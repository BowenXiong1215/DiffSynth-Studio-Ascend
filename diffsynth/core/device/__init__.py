from .npu_compatible_device import (
    empty_cache,
    get_available_device_type,
    get_device_name,
    parse_device_type,
    parse_nccl_backend,
    synchronize,
)
from .npu_compatible_device import IS_NPU_AVAILABLE, IS_CUDA_AVAILABLE
