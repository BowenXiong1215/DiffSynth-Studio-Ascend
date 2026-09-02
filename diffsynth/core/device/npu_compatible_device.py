import importlib.util
import os
from typing import Any, Optional, Union

import torch


def is_torch_npu_available() -> bool:
    """Return whether torch_npu can be imported without importing it eagerly."""
    return importlib.util.find_spec("torch_npu") is not None


# torch_npu registers torch.npu as an import side effect. Checking torch.npu
# before importing torch_npu makes a clean Ascend environment look like CPU.
if is_torch_npu_available():
    try:
        import torch_npu  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        # OSError also covers a wheel/driver/CANN ABI mismatch. Keep package
        # imports usable on CPU so diagnostics can report the real environment.
        torch_npu = None
else:
    torch_npu = None


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = (
    torch_npu is not None
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)

if IS_NPU_AVAILABLE and hasattr(torch.npu, "config"):
    # Keep public tensor layouts for checkpoint and CUDA/NPU alignment runs.
    torch.npu.config.allow_internal_format = False


def _normalize_device_type(device: Optional[Union[str, torch.device]]) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device.type
    device = str(device).lower()
    for device_type in ("npu", "cuda", "cpu", "mps"):
        if device == device_type or device.startswith(f"{device_type}:"):
            return device_type
    return None


def get_device_type() -> str:
    """Return the selected device type, honoring DIFFSYNTH_DEVICE when set."""
    requested = _normalize_device_type(os.environ.get("DIFFSYNTH_DEVICE"))
    if requested is not None:
        if requested == "npu" and not IS_NPU_AVAILABLE:
            raise RuntimeError(
                "DIFFSYNTH_DEVICE=npu was requested, but torch_npu/NPU is not "
                "available. Check the torch, torch_npu, driver and CANN versions."
            )
        if requested == "cuda" and not IS_CUDA_AVAILABLE:
            raise RuntimeError("DIFFSYNTH_DEVICE=cuda was requested, but CUDA is not available.")
        return requested
    if IS_CUDA_AVAILABLE:
        return "cuda"
    if IS_NPU_AVAILABLE:
        return "npu"
    return "cpu"


def get_torch_device(device: Optional[Union[str, torch.device]] = None) -> Any:
    """Get torch.cuda/torch.npu for an explicit device or selected device."""
    device_name = _normalize_device_type(device) or get_device_type()
    namespace = getattr(torch, device_name, None)
    if namespace is None:
        raise RuntimeError(f"Device namespace torch.{device_name} is not available.")
    return namespace


def get_device_id(device: Optional[Union[str, torch.device]] = None) -> int:
    device_type = _normalize_device_type(device) or get_device_type()
    if isinstance(device, torch.device) and device.index is not None:
        return device.index
    if isinstance(device, str) and ":" in device:
        return int(device.rsplit(":", 1)[1])
    if device_type == "cpu":
        return 0
    return get_torch_device(device_type).current_device()


def get_device_name(device: Optional[Union[str, torch.device]] = None) -> str:
    device_type = _normalize_device_type(device) or get_device_type()
    return device_type if device_type == "cpu" else f"{device_type}:{get_device_id(device)}"


def synchronize(device: Optional[Union[str, torch.device]] = None) -> None:
    device_type = _normalize_device_type(device) or get_device_type()
    if device_type != "cpu":
        get_torch_device(device_type).synchronize()


def empty_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    device_type = _normalize_device_type(device) or get_device_type()
    if device_type != "cpu":
        get_torch_device(device_type).empty_cache()


def get_nccl_backend() -> str:
    return parse_nccl_backend(get_device_type())


def enable_high_precision_for_bf16() -> None:
    """Disable reduced-precision reductions where the backend exposes a switch."""
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if IS_NPU_AVAILABLE:
        matmul = getattr(torch.npu, "matmul", None)
        if matmul is not None:
            if hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = False
            if hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
                matmul.allow_bf16_reduced_precision_reduction = False


def parse_device_type(device):
    return _normalize_device_type(device) or "cpu"


def parse_nccl_backend(device_type):
    device_type = parse_device_type(device_type)
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    raise RuntimeError(f"No distributed communication backend for device type {device_type}.")


def get_available_device_type():
    return get_device_type()
