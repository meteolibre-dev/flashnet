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

# 
from torch.optim import Muon

from safetensors.torch import load_file

# Add project root to sys.path
project_root = os.path.abspath("/workspace/flashnet/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning_radar import MeteoLibreMapDataset
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred import (
    trainer_step,
    full_image_generation,
)

from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v17_mtg_europe_lightning_radar_shortcut']


def extend_input_channels(state_dict, old_sat_in_channels, new_sat_in_channels):
    """
    Transfer learning: extend the first conv layer weights to accept additional input channels.
    Also extends the output layer to account for increased output channels.
    Copies existing channel weights and initializes new channels with the mean of existing ones.
    """
    if new_sat_in_channels <= old_sat_in_channels:
        return state_dict
    
    kpi_in_channels = 1  # from config
    kpi_out_channels = 1  # from config
    
    for key in list(state_dict.keys()):
        # Extend input conv layer (patch_embed.proj)
        if 'jit.patch_embed.proj.weight' in key:
            old_weight = state_dict[key]
            new_weight = torch.zeros(
                old_weight.shape[0],
                new_sat_in_channels + kpi_in_channels,
                *old_weight.shape[2:]
            )
            new_weight[:, :old_sat_in_channels + kpi_in_channels] = old_weight
            new_weight[:, old_sat_in_channels + kpi_in_channels:] = old_weight[:, :1].mean(dim=1, keepdim=True)
            state_dict[key] = new_weight
            print(f"Transfer learning: extended {key} from {old_weight.shape} to {new_weight.shape}")
        
        # Extend output linear layer (final_layer.linear)
        if 'jit.final_layer.linear.weight' in key:
            old_weight = state_dict[key]
            old_out_ch = old_sat_in_channels + kpi_out_channels  # 18
            new_out_ch = new_sat_in_channels + kpi_out_channels  # 19
            
            patch_t, patch_h, patch_w = 1, 8, 8  # from config
            old_feat = old_out_ch * patch_t * patch_h * patch_w   # 18 * 64 = 1152
            new_feat = new_out_ch * patch_t * patch_h * patch_w   # 19 * 64 = 1216
            
            new_weight = torch.zeros(new_feat, old_weight.shape[1])
            new_weight[:old_feat] = old_weight
            # Initialize new output channel with ZEROS so old outputs remain unchanged
            new_weight[old_feat:] = 0
            state_dict[key] = new_weight
            print(f"Transfer learning: extended {key} from {old_weight.shape} to {new_weight.shape}")
    

        # Extend output linear layer bias
        if 'jit.final_layer.linear.bias' in key:
            old_bias = state_dict[key]
            old_out_ch = old_sat_in_channels + kpi_out_channels  # 18
            new_out_ch = new_sat_in_channels + kpi_out_channels  # 19
            
            patch_t, patch_h, patch_w = 1, 8, 8
            old_feat = old_out_ch * patch_t * patch_h * patch_w
            new_feat = new_out_ch * patch_t * patch_h * patch_w
            
            new_bias = torch.zeros(new_feat)
            new_bias[:old_feat] = old_bias
            # Initialize new output channel bias with ZERO
            new_bias[old_feat:] = 0
            state_dict[key] = new_bias
            print(f"Transfer learning: extended {key} from {old_bias.shape} to {new_bias.shape}")

    return state_dict


def main():
    # Initialize Accelerator with bfloat16 precision and logging
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
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
        "radar_finetune_" + id_run, config=hps
    )

    # Initialize dataset
    dataset = MeteoLibreMapDataset(
        localrepo=params['dataset_path'],
        cache_size=10,
        seed=seed,
        nb_temporal=7
    )

    # Initialize DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,  # os.cpu_count() // 2,  # Use half the available CPUs
        pin_memory=True,
    )

    # Initialize model
    model_params = params["model"]

    def get_grouped_params(model):
        """Splits params into 2D (for Muon) and others (for AdamW)."""
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return muon_params, adamw_params

    class CombinedOptimizer:
        """Wrapper to make a list of optimizers behave like a single one."""
        def __init__(self, optimizers):
            self.optimizers = optimizers
        
        def step(self):
            for opt in self.optimizers:
                opt.step()
                
        def zero_grad(self):
            for opt in self.optimizers:
                opt.zero_grad()

        # Optional: proxies for state dicts if needed for checkpointing
        def state_dict(self):
            return [opt.state_dict() for opt in self.optimizers]

        def load_state_dict(self, state_dicts):
            for opt, state in zip(self.optimizers, state_dicts):
                opt.load_state_dict(state)

    if params["model_type"] == "jit":

        print("Jit model")
        model = DualJiT3D(**model_params)

        model = torch.compile(model)

        # here we 

        # Split params: Muon only accepts strictly 2D tensors
        muon_params, adamw_params = get_grouped_params(model)
        
        # 1. Muon for Transformer Internals (Matrices)
        # Note: Adjust momentum/nesterov args as per your Heavyball version if needed
        opt_muon = Muon(muon_params, lr=learning_rate, momentum=0.95, weight_decay=0.1)
        
        # 2. AdamW for Conv3d, Embeddings, Norms, Biases
        # Usually AdamW needs a lower LR than Muon
        opt_adam = torch.optim.AdamW(adamw_params, lr=learning_rate / 3, weight_decay=0.01)
        
        # Create a list for Accelerate
        optimizer = [opt_muon, opt_adam]
    else:
        model = DualUNet3DFiLM(**model_params)
        optimizer = ForeachSOAP(model.parameters(), lr=learning_rate, foreach=False, warmup_steps=100)

    # model_path = "models_world_shortcut/model_v16_mtg_world_lightning_shortcut_e120.safetensors"
    # #model_path = "models/epoch_15_mtg_meteofrance_.safetensors"
    # state_dict = load_file(model_path)
    
    # #### Transfer learning: extend the first conv layer to accept +1 radar channel
    # old_sat_in_channels = 17
    # new_sat_in_channels = model_params["sat_in_channels"]  # 18
    
    # state_dict = extend_input_channels(state_dict, old_sat_in_channels, new_sat_in_channels)
    # #### end transfert learning

    # model.load_state_dict(state_dict)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if isinstance(optimizer, list):
        optimizer = CombinedOptimizer(optimizer)

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
                generated_sample = generated_images[0, 17]  # Shape: (1, H, W)
                target_sample = x_target[0, 17].cpu()  # Shape: (1, H, W)

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
