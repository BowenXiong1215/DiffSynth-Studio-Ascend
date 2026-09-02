import importlib.util
import pathlib
import sys
import types
import unittest

import torch


def load_wan_video_dit_for_unit_test():
    """Load the target file without importing DiffSynth's optional media stack."""
    package_names = ["_rope_test_pkg", "_rope_test_pkg.models", "_rope_test_pkg.core"]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    camera = types.ModuleType("_rope_test_pkg.models.wan_video_camera_controller")
    camera.SimpleAdapter = object
    sys.modules[camera.__name__] = camera

    gradient = types.ModuleType("_rope_test_pkg.core.gradient")
    gradient.gradient_checkpoint_forward = lambda *args, **kwargs: None
    sys.modules[gradient.__name__] = gradient

    wantodance = types.ModuleType("_rope_test_pkg.models.wantodance")
    wantodance.WanToDanceRotaryEmbedding = object
    wantodance.WanToDanceMusicEncoderLayer = object
    sys.modules[wantodance.__name__] = wantodance

    path = pathlib.Path(__file__).parents[1] / "diffsynth/models/wan_video_dit.py"
    spec = importlib.util.spec_from_file_location("_rope_test_pkg.models.wan_video_dit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wan_video_dit = load_wan_video_dit_for_unit_test()
prepare_rope_freqs_for_real = wan_video_dit.prepare_rope_freqs_for_real
rope_apply_real = wan_video_dit.rope_apply_real


class WanRopeNpuLayoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.seq_len = 5
        self.num_heads = 3
        self.head_dim = 8
        angles = torch.randn(self.seq_len, 1, self.head_dim // 2)
        self.freqs_complex = torch.polar(torch.ones_like(angles), angles)

    def test_complex_freqs_are_converted_to_real_pairs(self):
        actual = prepare_rope_freqs_for_real(self.freqs_complex, "cpu")
        expected = torch.view_as_real(self.freqs_complex).float()
        self.assertEqual(actual.shape, (self.seq_len, 1, self.head_dim // 2, 2))
        self.assertFalse(torch.is_complex(actual))
        torch.testing.assert_close(actual, expected)

    def test_real_pair_input_is_idempotent(self):
        real_pairs = torch.view_as_real(self.freqs_complex).float()
        actual = prepare_rope_freqs_for_real(real_pairs, "cpu")
        torch.testing.assert_close(actual, real_pairs)

    def test_ambiguous_real_layout_fails_before_unpacking(self):
        bad_freqs = torch.randn(self.seq_len, 1, self.head_dim // 2)
        with self.assertRaisesRegex(ValueError, "final dimension of 2"):
            prepare_rope_freqs_for_real(bad_freqs, "cpu")

    def test_real_rope_matches_complex_reference(self):
        x = torch.randn(2, self.seq_len, self.num_heads * self.head_dim)
        actual = rope_apply_real(x, self.freqs_complex, self.num_heads)

        x_heads = x.reshape(2, self.seq_len, self.num_heads, self.head_dim)
        x_complex = torch.view_as_complex(
            x_heads.double().reshape(2, self.seq_len, self.num_heads, -1, 2)
        )
        expected = torch.view_as_real(x_complex * self.freqs_complex).flatten(2).to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
