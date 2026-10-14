"""Run FlashNet inference locally: local H5 input + local checkpoint, no GCP.

This bypasses the GCP-wired CLI pipeline in main.py and calls the
InferenceEngine directly, so nothing is downloaded from / uploaded to
Google Cloud Storage.

Usage (from the repo root):

    python backend/run_local_inference.py \
        --data  data/2026-07-25_21-50_europe_full.h5 \
        --model models/checkpoint.safetensors \
        --output inference_out/2026-07-25_21-50

Optional knobs:
    --forecast-steps N    number of forecast steps (default 18)
    --nb-forecast N       frames per forecast chunk (default 3)
    --denoising-steps N   NFE budget per chunk (default 128)
    --sampler euler|heun  ODE sampler (default euler)
    --batch-size N        patch batch size (default 64)
"""

import argparse
import sys
from pathlib import Path

# Make "python backend/run_local_inference.py" work from anywhere by putting
# the repo root (parent of backend/) on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.inference_engine import InferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlashNet local inference (no GCP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to input H5 file")
    parser.add_argument("--model", required=True, help="Path to model .safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for forecast TIFFs")
    parser.add_argument("--forecast-steps", type=int, default=18)
    parser.add_argument("--nb-forecast", type=int, default=3)
    parser.add_argument("--denoising-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sampler", choices=["euler", "heun"], default="euler")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for label, path in [("data", args.data), ("model", args.model)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    engine = InferenceEngine(
        model_path=args.model,
        model_type="jit",
        patch_size=128,
        denoising_steps=args.denoising_steps,
        batch_size=args.batch_size,
        context_frames=4,
        use_residual=False,
        interpolation="linear",
        sampler=args.sampler,
        device=args.device,
    )

    result = engine.run_inference(
        data_path=args.data,
        output_dir=args.output,
        forecast_steps=args.forecast_steps,
        nb_forecast=args.nb_forecast,
    )

    out_files = sorted(Path(args.output).glob("*.tiff"))
    print(f"\nInference finished: status={getattr(result.status, 'value', result.status)}")
    print(f"Output directory: {args.output}")
    print(f"Wrote {len(out_files)} TIFF files:")
    for f in out_files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
