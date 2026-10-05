"""Small deterministic sanity tests for the coordinate-preserving RMT module."""

import unittest

import numpy as np

from rmt_protection import generate_orthogonal_matrices, transform_rgb_image


class RMTProtectionTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:256, :256]
        self.image = np.stack(
            [xx / 255.0, yy / 255.0, ((xx + yy) % 256) / 255.0], axis=-1
        ).astype(np.float32)
        self.mask = ((xx - 128) ** 2 + (yy - 128) ** 2 < 40 ** 2).astype(np.uint8)

    def test_shape_finiteness_and_determinism(self):
        first = transform_rgb_image(self.image, 16, 0.05, "sample-a.png", 777)
        np.random.seed(999)  # global/training RNG state must not affect RMT
        second = transform_rgb_image(self.image, 16, 0.05, "sample-a.png", 777)
        self.assertEqual(first.shape, (256, 256, 3))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_training_seed_does_not_enter_transformation(self):
        outputs = [
            transform_rgb_image(self.image, 16, 0.05, "sample-a.png", 777)
            for _training_seed in (42, 101, 202, 303)
        ]
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)

    def test_fixed_matrices_are_orthogonal_and_position_specific(self):
        for block_size in (8, 16, 32):
            matrices = generate_orthogonal_matrices(256, block_size, 777)
            matrices_again = generate_orthogonal_matrices(256, block_size, 777)
            self.assertEqual(matrices.shape, ((256 // block_size) ** 2, block_size, block_size))
            np.testing.assert_array_equal(matrices, matrices_again)
            np.testing.assert_allclose(
                np.einsum("ji,jk->ik", matrices[0], matrices[0]),
                np.eye(block_size), atol=1e-12
            )
            self.assertFalse(np.array_equal(matrices[0], matrices[1]))
        self.assertEqual([256 % size for size in (8, 16, 32)], [0, 0, 0])

    def test_zero_noise_and_key_changes(self):
        zero_noise = transform_rgb_image(self.image, 16, 0.0, "sample-a.png", 777)
        zero_noise_again = transform_rgb_image(self.image, 16, 0.0, "sample-a.png", 777)
        np.testing.assert_array_equal(zero_noise, zero_noise_again)
        changed_key = transform_rgb_image(self.image, 16, 0.05, "sample-a.png", 778)
        baseline_key = transform_rgb_image(self.image, 16, 0.05, "sample-a.png", 777)
        self.assertFalse(np.array_equal(changed_key, baseline_key))

    def test_block_boundaries_and_mask_are_preserved(self):
        original_mask = self.mask.copy()
        changed_image = self.image.copy()
        changed_image[:16, :16, :] = 0.0
        before = transform_rgb_image(self.image, 16, 0.05, "sample-a.png", 777)
        after = transform_rgb_image(changed_image, 16, 0.05, "sample-a.png", 777)
        # An input block cannot affect pixels in any other block position.
        np.testing.assert_array_equal(before[16:, :, :], after[16:, :, :])
        np.testing.assert_array_equal(before[:16, 16:, :], after[:16, 16:, :])
        np.testing.assert_array_equal(self.mask, original_mask)

    def test_sample_specific_noise_is_deterministic(self):
        a = transform_rgb_image(self.image, 8, 0.10, "sample-a.png", 777)
        b = transform_rgb_image(self.image, 8, 0.10, "sample-b.png", 777)
        a_again = transform_rgb_image(self.image, 8, 0.10, "sample-a.png", 777)
        np.testing.assert_array_equal(a, a_again)
        self.assertFalse(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
