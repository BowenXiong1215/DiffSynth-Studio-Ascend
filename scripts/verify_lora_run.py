import argparse
import csv
import math
from pathlib import Path

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description="Verify a DiffSynth LoRA training output.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    checkpoints = sorted(args.output_dir.glob("*.safetensors"))
    if not checkpoints:
        raise SystemExit(f"No LoRA checkpoint in {args.output_dir}")
    for checkpoint in checkpoints:
        if checkpoint.stat().st_size == 0:
            raise SystemExit(f"Empty checkpoint: {checkpoint}")
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if not keys:
                raise SystemExit(f"Checkpoint contains no tensors: {checkpoint}")
            for key in keys:
                if not tensors.get_tensor(key).isfinite().all().item():
                    raise SystemExit(f"Non-finite tensor: {checkpoint}:{key}")

    loss_path = args.output_dir / "loss.csv"
    if not loss_path.is_file() or loss_path.stat().st_size == 0:
        raise SystemExit(f"Missing loss log: {loss_path}")
    with loss_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    losses = [float(row["value"]) for row in rows if row.get("key") == "loss"]
    if not losses:
        raise SystemExit(f"No loss values in {loss_path}")
    if not all(math.isfinite(value) for value in losses):
        raise SystemExit(f"Non-finite loss in {loss_path}")

    print(f"LoRA checkpoints: {len(checkpoints)}")
    print(f"Loss records: {len(losses)}")
    print(f"Loss range: {min(losses):.8g} .. {max(losses):.8g}")
    print("LoRA run verification: PASS")


if __name__ == "__main__":
    main()
