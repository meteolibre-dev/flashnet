"""
Training script for MeteoLibre using Hugging Face Accelerate with Rectified Flow (shortcut version)
This script trains a rectified flow model using the MeteoLibreMapDataset and UNet_DCAE_3D.
"""

import sys
import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from datetime import datetime
import yaml
from contextlib import contextmanager, nullcontext

from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_file

# 
#from torch.optim import Muon

from safetensors.torch import load_file

# Add project root to sys.path
project_root = os.path.abspath("/workspace/flashnet/")
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_mtg_lightning_radar_cg import MeteoLibreMapDataset
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred_blur_v2 import (
    trainer_step,
    full_image_generation,
)

from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v24_mtg_europe_lightning_radar_shortcut']


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Smooths the high-frequency weight oscillations that bf16 + an aggressive LR
    induce late in training. EMA does NOT lower the train loss (that is computed
    with the raw weights) — it improves SAMPLE quality, noticeably for few-step
    sampling and autoregressive rollouts where weight noise compounds across
    steps.

    Usage:
        ema = EMAModel(model.parameters(), decay=0.9999)
        ema.update(model.parameters())      # after each optimizer.step()
        with ema.swap(model.parameters()):  # eval / save with EMA weights
            ...

    - shadow params stored in fp32 for accurate accumulation, cast back to each
      param's native dtype when swapped in.
    - warmup: effective decay = min(decay, (1+step)/(10+step)). Early on (weights
      moving fast) the EMA tracks the raw model closely; as training stabilizes
      it ramps up to `decay` for strong smoothing. Shadow is initialized to the
      params themselves, so no Adam-style bias correction is needed.
    - decay=0.9999 ~= half-life of ~7000 updates (~6 epochs at ~1172 optimizer
      steps/epoch with this config).
    """

    def __init__(self, parameters, decay=0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = [p.detach().to(torch.float32).clone() for p in parameters]

    @torch.no_grad()
    def update(self, parameters):
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for s, p in zip(self.shadow_params, parameters):
            s.mul_(d).add_(p.detach().to(torch.float32), alpha=1.0 - d)

    @contextmanager
    def swap(self, parameters):
        """Temporarily copy EMA weights into `parameters` (in-place on .data),
        yield, then restore the originals."""
        backup = [p.detach().clone() for p in parameters]
        try:
            for p, s in zip(parameters, self.shadow_params):
                p.data.copy_(s.to(p.dtype))
            yield
        finally:
            for p, b in zip(parameters, backup):
                p.data.copy_(b)


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
    seed = params['seed'] + int(random.random() * 1000)
    residual = bool(params.get('residual', True))
    sigma_noise_input = params['sigma_noise_input']

    print("sigma_noise_input: ", sigma_noise_input)

    ema_enabled = bool(params.get('ema_enabled', True))
    ema_decay = float(params.get('ema_decay', 0.9999))
    print(f"EMA: enabled={ema_enabled}, decay={ema_decay}")

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
        shuffle=False,  # file order shuffled once in Dataset.__init__ (shared
                        # across workers) for 100% coverage + parquet locality.
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

        model_path = "models/checkpoint.safetensors"
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)

        model = torch.compile(model) 

        # Split params: Muon only accepts strictly 2D tensors
        muon_params, adamw_params = get_grouped_params(model)
        
        # 1. Muon for Transformer Internals (Matrices)
        # Note: Adjust momentum/nesterov args as per your Heavyball version if needed
        # opt_muon = Muon(muon_params, lr=learning_rate, momentum=0.95, weight_decay=0.1)
        opt_muon = torch.optim.AdamW(muon_params, lr=learning_rate, weight_decay=0.01)
        # 2. AdamW for Conv3d, Embeddings, Norms, Biases
        # Usually AdamW needs a lower LR than Muon
        opt_adam = torch.optim.AdamW(adamw_params, lr=learning_rate / 3, weight_decay=0.01)
        
        # Create a list for Accelerate
        optimizer = [opt_muon, opt_adam]
    else:
        exit()

    #model_path = "models_world_shortcut/model_v16_mtg_world_lightning_shortcut_e120.safetensors"


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if isinstance(optimizer, list):
        optimizer = CombinedOptimizer(optimizer)

    # ---- EMA of weights (improves sample quality, not train loss) ----
    ema = None
    if ema_enabled:
        _raw = accelerator.unwrap_model(model)
        _raw = getattr(_raw, '_orig_mod', _raw)
        ema = EMAModel(list(_raw.parameters()), decay=ema_decay)
        print(f"EMA tracking {len(ema.shadow_params)} param tensors")

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
                    model, batch, device, parametrization=PARAMETRIZATION, interpolation=INTERPOLATION, sigma=sigma_noise_input, use_residual=residual,
                    noise_rho=float(params.get('noise_rho', 0.90)),
                    temporal_weight_scale=float(params.get('temporal_weight_scale', 1.0)),
                    context_spec_noise=float(params.get('context_spec_noise', 0.0)),
                    context_spec_noise_prob=float(params.get('context_spec_noise_prob', 0.5)),
                    context_spec_noise_frame_ramp=float(params.get('context_spec_noise_frame_ramp', 0.0)),
                    d_x0_blur_prob=float(params.get('d_x0_blur_prob', 0.0)),
                    d_x0_blur_scale=float(params.get('d_x0_blur_scale', 1.0)),
                )

                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # EMA update — once per real optimizer step (grad-accum boundary)
                if ema is not None and accelerator.sync_gradients:
                    _raw = accelerator.unwrap_model(model)
                    _raw = getattr(_raw, '_orig_mod', _raw)
                    ema.update(_raw.parameters())

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
                # eval on the raw (un-compiled) module with EMA weights swapped in
                eval_model = getattr(unwrapped_model, '_orig_mod', unwrapped_model)
                swap_ctx = ema.swap(list(eval_model.parameters())) if ema is not None else nullcontext()
                with swap_ctx:
                    generated_images, x_target = full_image_generation(
                        eval_model,
                        batch,
                        steps=128,
                        device=accelerator.device,
                        parametrization=PARAMETRIZATION,
                        interpolation=INTERPOLATION,
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
                model_to_save = getattr(unwrapped_model, '_orig_mod', unwrapped_model)

                save_path = f"{MODEL_DIR}epoch_{epoch + 1}_mtg_meteofrance_.safetensors"
                save_path_check = f"{MODEL_DIR}checkpoint.safetensors"
                save_path_raw = f"{MODEL_DIR}epoch_{epoch + 1}_mtg_meteofrance__raw.safetensors"

                os.makedirs(MODEL_DIR, exist_ok=True)

                # Always keep the raw (training) weights
                save_file(model_to_save.state_dict(), save_path_raw)

                # EMA weights are the primary deploy checkpoint (sharper / more
                # stable sampling). Falls back to raw if EMA disabled.
                if ema is not None:
                    with ema.swap(list(model_to_save.parameters())):
                        save_file(model_to_save.state_dict(), save_path)
                        save_file(model_to_save.state_dict(), save_path_check)
                    accelerator.print(f"Model saved (EMA) to {save_path}")
                else:
                    save_file(model_to_save.state_dict(), save_path)
                    save_file(model_to_save.state_dict(), save_path_check)
                    accelerator.print(f"Model saved to {save_path}")


        accelerator.wait_for_everyone()


    # Save the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        model_to_save = getattr(unwrapped_model, '_orig_mod', unwrapped_model)
        if ema is not None:
            with ema.swap(list(model_to_save.parameters())):
                save_file(model_to_save.state_dict(), f"{MODEL_DIR}final_mtg_meteofrance_ema.safetensors")
            print("Training complete. EMA model saved.")
        else:
            torch.save(model.state_dict(), "meteolibre_model_rectified_flow.pth")
            print("Training complete. Model saved to meteolibre_model_rectified_flow.pth")


if __name__ == "__main__":
    main()
