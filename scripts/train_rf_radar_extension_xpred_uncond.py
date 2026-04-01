"""
Training script for unconditional rectified flow weather generation.
Generates frames directly from noise without any conditioning on context frames.
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
from torch.optim import Muon
from safetensors.torch import load_file

# Add project root to sys.path
project_root = os.path.abspath("/workspace/flashnet/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning_radar import MeteoLibreMapDataset
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred_uncond import (
    trainer_step,
    full_image_generation,
)
from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v21_mtg_europe_lightning_radar_shortcut']


def main():

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        log_with="tensorboard",
        project_dir=".",
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device

    LOG_EVERY_N_STEPS = params['log_every_n_steps']
    SAVE_EVERY_N_EPOCHS = params['save_every_n_epochs']
    MODEL_DIR = params['model_dir']
    PARAMETRIZATION = params['parametrization']
    INTERPOLATION = params.get('interpolation', 'linear')
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']
    seed = params['seed']
    sigma_noise_input = params['sigma_noise_input']

    print("sigma_noise_input: ", sigma_noise_input)

    gradient_clip_value = params['gradient_clip_value']
    id_run = str(datetime.utcnow())[:19]
    set_seed(seed)

    hps = {"batch_size": batch_size, "learning_rate": learning_rate}

    accelerator.init_trackers("radar_uncond_" + id_run, config=hps)

    dataset = MeteoLibreMapDataset(
        localrepo=params['dataset_path'],
        cache_size=10,
        seed=seed,
        nb_temporal=7,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model_params = params["model"]

    def get_grouped_params(model):
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
        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for opt in self.optimizers:
                opt.step()

        def zero_grad(self):
            for opt in self.optimizers:
                opt.zero_grad()

        def state_dict(self):
            return [opt.state_dict() for opt in self.optimizers]

        def load_state_dict(self, state_dicts):
            for opt, state in zip(self.optimizers, state_dicts):
                opt.load_state_dict(state)

    if params["model_type"] == "jit":
        print("Jit model")
        model = DualJiT3D(**model_params)
        model = torch.compile(model)

        muon_params, adamw_params = get_grouped_params(model)
        opt_muon = Muon(muon_params, lr=learning_rate, momentum=0.95, weight_decay=0.1)
        opt_adam = torch.optim.AdamW(adamw_params, lr=learning_rate / 3, weight_decay=0.01)
        optimizer = [opt_muon, opt_adam]
    else:
        exit()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if isinstance(optimizer, list):
        optimizer = CombinedOptimizer(optimizer)

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:

            with accelerator.accumulate(model):
                loss, loss_sat, loss_kpi = trainer_step(
                    model, batch, device,
                    parametrization=PARAMETRIZATION,
                    interpolation=INTERPOLATION,
                    sigma=sigma_noise_input,
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % LOG_EVERY_N_STEPS == 0:
                    if accelerator.is_main_process:
                        accelerator.log({"Loss/train_trained": loss.item()}, step=global_step)
                        accelerator.log({"Loss_sat/train_trained": loss_sat.item()}, step=global_step)
                        accelerator.log({"Loss_kpi/train_trained": loss_kpi.item()}, step=global_step)

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        accelerator.log({"Loss/train": avg_loss}, step=epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)
                generated_images, x_target = full_image_generation(
                    unwrapped_model,
                    batch,
                    steps=16,
                    device=accelerator.device,
                    parametrization=PARAMETRIZATION,
                    interpolation=INTERPOLATION,
                )

                generated_sample = generated_images[0, 17]
                target_sample = x_target[0, 17].cpu()

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
                    tb_tracker.writer.add_image("Generated vs Target (normalized)", grid_normalized, epoch)

        if (epoch) % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = f"{MODEL_DIR}epoch_{epoch + 1}_uncond_.safetensors"
                os.makedirs(MODEL_DIR, exist_ok=True)
                model_to_save = getattr(unwrapped_model, '_orig_mod', unwrapped_model)
                save_file(model_to_save.state_dict(), save_path)
                accelerator.print(f"Model saved to {save_path}")

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), "meteolibre_model_uncond_rectified_flow.pth")
        print("Training complete. Model saved to meteolibre_model_uncond_rectified_flow.pth")


if __name__ == "__main__":
    main()
