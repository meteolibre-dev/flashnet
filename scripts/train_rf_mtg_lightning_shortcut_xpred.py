"""
Training script for MeteoLibre using Hugging Face Accelerate with Rectified Flow (shortcut version)
This script trains a rectified flow model using the MeteoLibreMapDataset and UNet_DCAE_3D.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from datetime import datetime
import yaml

from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_file

# custom optimizer TO REMOVE ?
from heavyball import ForeachSOAP, ForeachMuon

from safetensors.torch import load_file

# Add project root to sys.path
project_root = os.path.abspath("/workspace/flashnet/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning_v2 import MeteoLibreMapDataset
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred import (
    trainer_step,
    full_image_generation,
)

from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v13_mtg_world_lightning_shortcut']

def main():
    # Initialize Accelerator with bfloat16 precision and logging
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=".",
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device

    # Load hyperparameters from config
    LOG_EVERY_N_STEPS = params['log_every_n_steps']
    SAVE_EVERY_N_EPOCHS = params['save_every_n_epochs']
    MODEL_DIR = params['model_dir']
    PARAMETRIZATION = params['parametrization']
    INTERPOLATION = params.get('interpolation', 'linear')
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']
    seed = params['seed']
    residual = bool(params.get('residual', True))
    sigma_noise_input = params['sigma_noise_input']
    gradient_clip_value = params['gradient_clip_value']
    id_run = str(datetime.utcnow())[:19]
    # Set seed for reproducibility
    set_seed(seed)

    hps = {"batch_size": batch_size, "learning_rate": learning_rate}
    print("residual is :", residual)

    accelerator.init_trackers(
        "lightning_shortcut-eps-prediction-training-rectified-flow_" + id_run, config=hps
    )

    # Initialize dataset
    dataset = MeteoLibreMapDataset(
        localrepo=params['dataset_path'],
        cache_size=4,
        seed=seed,
        nb_temporal=7
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # os.cpu_count() // 2,  # Use half the available CPUs
        pin_memory=True,
    )

    # Initialize model
    model_params = params["model"]
    model = DualUNet3DFiLM(**model_params)

    # model_path = "models/epoch_6_mtg_meteofrance_.safetensors"
    # state_dict = load_file(model_path)
    
    # model.load_state_dict(state_dict)

    # Initialize optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = ForeachSOAP(model.parameters(), lr=learning_rate, foreach=False, warmup_steps=100)
    #optimizer = torch.optim.Muon(model.parameters(), lr=learning_rate)

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:

            # Perform training step
            with accelerator.accumulate(model):
                loss, loss_sat, loss_kpi = trainer_step(
                    model, batch, device, parametrization=PARAMETRIZATION, interpolation=INTERPOLATION, sigma=sigma_noise_input, use_residual=residual
                )

                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % LOG_EVERY_N_STEPS == 0:
                    if accelerator.is_main_process:
                        accelerator.log(
                            {"Loss/train_trained": loss.item()},
                            step=global_step,
                        )

                        accelerator.log(
                            {"Loss_sat/train_trained": loss_sat.item()},
                            step=global_step,
                        )

                        accelerator.log(
                            {"Loss_kpi/train_trained": loss_kpi.item()},
                            step=global_step,
                        )

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Log to Accelerate
        accelerator.log({"Loss/train": avg_loss}, step=epoch)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            with torch.no_grad():
                # x_target = normalize(x_target, device)

                unwrapped_model = accelerator.unwrap_model(model)
                generated_images, x_target = full_image_generation(
                    unwrapped_model,
                    batch,
                    device=accelerator.device,
                    parametrization=PARAMETRIZATION,
                    use_residual=residual
                )

                # Select one channel and one batch item for visualization
                generated_sample = generated_images[0, 11]  # Shape: (1, H, W)
                target_sample = x_target[0, 11].cpu()  # Shape: (1, H, W)

                all_frames = torch.cat([generated_sample, target_sample], dim=0) / 8.0
                all_frames = all_frames.clamp(-10, 10)

                grid = make_grid(all_frames.unsqueeze(1), nrow=2)
                grid_normalized = make_grid(
                    (all_frames.unsqueeze(1) - all_frames.min())
                    / (all_frames.max() - all_frames.min()),
                    nrow=2,
                )

                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker:
                    tb_tracker.writer.add_image("Generated vs Target", grid, epoch)
                    tb_tracker.writer.add_image(
                        "Generated vs Target (normalized)", grid_normalized, epoch
                    )
                


        # This part for saving the model was already correct
        if (epoch) % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                # Save the EMA model's state dictionary
                save_path = f"{MODEL_DIR}epoch_{epoch + 1}_mtg_meteofrance_.safetensors"
                os.makedirs(MODEL_DIR, exist_ok=True)
                save_file(unwrapped_model.state_dict(), save_path)
                accelerator.print(f"Model saved to {save_path}")

        accelerator.wait_for_everyone()


    # Save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), "meteolibre_model_rectified_flow.pth")
        print("Training complete. Model saved to meteolibre_model_rectified_flow.pth")


if __name__ == "__main__":
    main()
