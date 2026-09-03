"""Compare named CUDA and NPU tensors saved as torch dictionaries."""

import argparse
import math

import torch


def flatten_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value.detach().cpu().float().reshape(-1)
    elif isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_tensors(value[key], child)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten_tensors(item, child)


def load_tensor_map(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return dict(flatten_tensors(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="CUDA/reference .pt file")
    parser.add_argument("candidate", help="NPU/candidate .pt file")
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    args = parser.parse_args()

    reference = load_tensor_map(args.reference)
    candidate = load_tensor_map(args.candidate)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        unexpected = sorted(candidate.keys() - reference.keys())
        raise SystemExit(f"Tensor key mismatch. missing={missing}, unexpected={unexpected}")

    failed = False
    print(f"{'tensor':60} {'cosine':>12} {'rel_l2':>12} {'max_abs':>12}")
    for name in reference:
        left, right = reference[name], candidate[name]
        if left.shape != right.shape:
            print(f"{name}: shape mismatch {tuple(left.shape)} != {tuple(right.shape)}")
            failed = True
            continue
        max_abs = (left - right).abs().max().item() if left.numel() else 0.0
        denominator = left.norm().item()
        relative_l2 = (left - right).norm().item() / max(denominator, 1e-12)
        if left.numel() == 0:
            cosine = 1.0
        elif left.norm().item() == 0 or right.norm().item() == 0:
            cosine = 1.0 if torch.equal(left, right) else 0.0
        else:
            cosine = torch.nn.functional.cosine_similarity(left, right, dim=0).item()
        print(f"{name[:60]:60} {cosine:12.8f} {relative_l2:12.6g} {max_abs:12.6g}")
        if not math.isfinite(cosine) or cosine < args.min_cosine or relative_l2 > args.max_relative_l2:
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
