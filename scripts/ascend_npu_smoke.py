"""Fail-fast environment and core-operator probe for DiffSynth on Ascend."""

import sys

import torch


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains NaN or Inf")


def main():
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise RuntimeError("torch_npu import failed; check the torch/CANN/driver matrix") from exc

    from diffsynth.core.device import get_available_device_type, synchronize

    device_type = get_available_device_type()
    if device_type != "npu":
        raise RuntimeError(f"Expected an NPU runtime, selected device is {device_type!r}")

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"python={sys.version.split()[0]} torch={torch.__version__} device={device}")

    q = torch.randn(1, 4, 64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    attention = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    attention.float().square().mean().backward()
    assert_finite("sdpa.output", attention)
    assert_finite("sdpa.q_grad", q.grad)

    conv = torch.nn.Conv3d(4, 8, 3, padding=1, device=device, dtype=torch.bfloat16)
    video = torch.randn(1, 4, 3, 16, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    conv_output = conv(video)
    conv_output.float().mean().backward()
    assert_finite("conv3d.output", conv_output)
    assert_finite("conv3d.input_grad", video.grad)

    parameter = torch.nn.Parameter(torch.ones(16, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        parameter.square().mean().backward()
        optimizer.step()
    assert_finite("adamw.parameter", parameter)
    synchronize(device)
    print("PASS: BF16 SDPA, BF16 Conv3D and FP32 AdamW forward/backward")


if __name__ == "__main__":
    main()
