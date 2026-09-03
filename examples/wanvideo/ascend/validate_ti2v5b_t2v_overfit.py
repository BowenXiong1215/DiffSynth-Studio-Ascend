"""Compare base and LoRA TI2V-5B T2V outputs on one training sample."""

import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diffsynth.core.device import get_available_device_type
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


DEFAULT_ROOT = Path("/hpc-to-ds-0115/x00876811")
DEFAULT_REPO = DEFAULT_ROOT / "DiffSynth-Studio-Ascend-main"
DEFAULT_DATASET = (
    DEFAULT_REPO
    / "data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_ROOT
    / "outputs/ti2v-5b-t2v-overfit-49f/step-500.safetensors"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs/ti2v-5b-t2v-overfit-49f-validation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate target, base and LoRA videos for TI2V-5B T2V validation."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", default="metadata_overfit.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--grid-columns", type=int, default=7)
    return parser.parse_args()


def require_file(path, name):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{name} unavailable: {path}")


def load_first_sample(dataset_dir, metadata_path, height, width, num_frames):
    with metadata_path.open(encoding="utf-8-sig") as file:
        row = next(csv.DictReader(file))

    video_path = Path(row["video"])
    if not video_path.is_absolute():
        video_path = dataset_dir / video_path
    require_file(video_path, "Training video")

    source = VideoData(video_file=str(video_path), height=height, width=width)
    if len(source) < num_frames:
        raise ValueError(
            f"Training video contains {len(source)} frames, expected at least {num_frames}"
        )
    frames = [source[index] for index in range(num_frames)]
    return str(row["prompt"]), video_path, frames


def load_pipeline(device):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
            ),
        ],
    )


def save_frame_grid(frames, path, columns):
    if not frames:
        raise ValueError("Cannot create a grid from an empty frame list")
    if columns <= 0:
        raise ValueError("grid-columns must be positive")

    frames = [frame.convert("RGB") for frame in frames]
    frame_width, frame_height = frames[0].size
    rows = (len(frames) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * frame_width, rows * frame_height),
        color=(0, 0, 0),
    )

    for index, frame in enumerate(frames):
        if frame.size != (frame_width, frame_height):
            frame = frame.resize((frame_width, frame_height))
        x = (index % columns) * frame_width
        y = (index // columns) * frame_height
        grid.paste(frame, (x, y))
        draw = ImageDraw.Draw(grid)
        label = f"#{index:02d}"
        draw.rectangle((x + 4, y + 4, x + 42, y + 20), fill=(0, 0, 0))
        draw.text((x + 7, y + 6), label, fill=(255, 255, 255))

    grid.save(path)


def main():
    args = parse_args()
    metadata_path = args.dataset_dir / args.metadata
    require_file(args.checkpoint, "LoRA checkpoint")
    require_file(metadata_path, "Metadata")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt, video_path, target_frames = load_first_sample(
        args.dataset_dir,
        metadata_path,
        args.height,
        args.width,
        args.num_frames,
    )
    device = get_available_device_type()

    print(f"Device: {device}")
    print(f"Video: {video_path}")
    print(f"Prompt: {prompt}")
    print(f"Checkpoint: {args.checkpoint}")

    target_path = args.output_dir / "target.mp4"
    base_path = args.output_dir / "base.mp4"
    lora_path = args.output_dir / "lora.mp4"
    target_grid_path = args.output_dir / "target_grid.png"
    base_grid_path = args.output_dir / "base_grid.png"
    lora_grid_path = args.output_dir / "lora_grid.png"
    save_video(target_frames, str(target_path), fps=args.fps, quality=8)
    save_frame_grid(target_frames, target_grid_path, args.grid_columns)

    pipe = load_pipeline(device)
    generation_args = {
        "prompt": prompt,
        "negative_prompt": "",
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "tiled": True,
    }

    print("Generating base.mp4...")
    base_frames = pipe(**generation_args)
    save_video(base_frames, str(base_path), fps=args.fps, quality=8)
    save_frame_grid(base_frames, base_grid_path, args.grid_columns)

    print("Loading LoRA...")
    pipe.load_lora(pipe.dit, str(args.checkpoint), alpha=args.lora_alpha)

    print("Generating lora.mp4...")
    lora_frames = pipe(**generation_args)
    save_video(lora_frames, str(lora_path), fps=args.fps, quality=8)
    save_frame_grid(lora_frames, lora_grid_path, args.grid_columns)

    print("Validation completed:")
    print(f"Target: {target_path}")
    print(f"Base:   {base_path}")
    print(f"LoRA:   {lora_path}")
    print(f"Target grid: {target_grid_path}")
    print(f"Base grid:   {base_grid_path}")
    print(f"LoRA grid:   {lora_grid_path}")


if __name__ == "__main__":
    main()
