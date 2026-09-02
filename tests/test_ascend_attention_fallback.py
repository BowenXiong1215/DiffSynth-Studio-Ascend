import unittest
from unittest import mock

import torch

from diffsynth.core.attention import attention


class AttentionFallbackTest(unittest.TestCase):
    def test_non_cuda_tensor_never_uses_cuda_flash_attention(self):
        query = torch.randn(1, 2, 8, 4)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        with mock.patch.object(attention, "ATTENTION_IMPLEMENTATION", "flash_attention_2"):
            with mock.patch.object(attention, "flash_attention_2", side_effect=AssertionError("CUDA path selected")):
                output = attention.attention_forward(query, key, value)
        self.assertEqual(output.shape, query.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
