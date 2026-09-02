import os
import unittest
from unittest import mock

import torch

from diffsynth.core.device import npu_compatible_device as device_backend


class DeviceBackendTest(unittest.TestCase):
    def test_parse_device_type(self):
        self.assertEqual(device_backend.parse_device_type("npu:7"), "npu")
        self.assertEqual(device_backend.parse_device_type("cuda:1"), "cuda")
        self.assertEqual(device_backend.parse_device_type(torch.device("cpu")), "cpu")
        self.assertEqual(device_backend.parse_device_type("not-a-device"), "cpu")

    def test_cpu_override(self):
        with mock.patch.dict(os.environ, {"DIFFSYNTH_DEVICE": "cpu"}):
            self.assertEqual(device_backend.get_device_type(), "cpu")
            self.assertEqual(device_backend.get_device_name(), "cpu")
            device_backend.synchronize()
            device_backend.empty_cache()

    def test_unavailable_npu_override_fails_early(self):
        if device_backend.IS_NPU_AVAILABLE:
            self.skipTest("This assertion is for CPU/CUDA CI")
        with mock.patch.dict(os.environ, {"DIFFSYNTH_DEVICE": "npu"}):
            with self.assertRaisesRegex(RuntimeError, "torch_npu/NPU is not available"):
                device_backend.get_device_type()


if __name__ == "__main__":
    unittest.main()
