import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.landmarks.normalizer import LandmarkNormalizer

class TestLandmarkNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = LandmarkNormalizer()

        # Create a valid synthetic right hand with 21 landmarks
        self.valid_right_hand = []
        for i in range(21):
            self.valid_right_hand.append({'x': i * 0.1, 'y': i * 0.2, 'z': i * 0.05})

        # Ensure non-zero scale (wrist to middle MCP, index 0 to 9)
        # distance = sqrt((0.9-0)^2 + (1.8-0)^2 + (0.45-0)^2) > 0

    def test_output_shape_and_dtype(self):
        """1 & 2. Output shape is exactly (63,) and dtype is float32."""
        valid, features, reason = self.normalizer.normalize(self.valid_right_hand, "Right")
        self.assertTrue(valid)
        self.assertEqual(features.shape, (63,))
        self.assertEqual(features.dtype, np.float32)

    def test_finite_values(self):
        """3. All values are finite."""
        valid, features, reason = self.normalizer.normalize(self.valid_right_hand, "Right")
        self.assertTrue(valid)
        self.assertTrue(np.isfinite(features).all())

        # Test with NaN
        bad_hand = list(self.valid_right_hand)
        bad_hand[5] = {'x': float('nan'), 'y': 0.1, 'z': 0.1}
        valid, features, reason = self.normalizer.normalize(bad_hand, "Right")
        self.assertFalse(valid)
        self.assertIn("non-finite", reason)

    def test_translation_invariance(self):
        """4. Translation invariance."""
        valid1, feat1, _ = self.normalizer.normalize(self.valid_right_hand, "Right")

        # Translate hand
        translated_hand = []
        for lm in self.valid_right_hand:
            translated_hand.append({'x': lm['x'] + 100.5, 'y': lm['y'] - 50.2, 'z': lm['z'] + 10.0})

        valid2, feat2, _ = self.normalizer.normalize(translated_hand, "Right")

        self.assertTrue(valid1 and valid2)
        np.testing.assert_allclose(feat1, feat2, rtol=1e-5, atol=1e-5)

    def test_scale_invariance(self):
        """5. Scale invariance."""
        valid1, feat1, _ = self.normalizer.normalize(self.valid_right_hand, "Right")

        # Scale hand from wrist (origin)
        scaled_hand = []
        for lm in self.valid_right_hand:
            scaled_hand.append({'x': lm['x'] * 5.0, 'y': lm['y'] * 5.0, 'z': lm['z'] * 5.0})

        valid2, feat2, _ = self.normalizer.normalize(scaled_hand, "Right")

        self.assertTrue(valid1 and valid2)
        np.testing.assert_allclose(feat1, feat2, rtol=1e-5, atol=1e-5)

    def test_zero_scale_handling(self):
        """6. Zero-scale handling."""
        # Create an invalid hand where wrist (0) and middle MCP (9) are identical
        zero_scale_hand = []
        for i in range(21):
            zero_scale_hand.append({'x': 1.0, 'y': 2.0, 'z': 3.0})

        valid, features, reason = self.normalizer.normalize(zero_scale_hand, "Right")
        self.assertFalse(valid)
        self.assertIsNone(features)
        self.assertIn("Zero or near-zero scale", reason)

    def test_wrist_is_origin(self):
        """7. Wrist becomes the origin after normalization."""
        valid, features, _ = self.normalizer.normalize(self.valid_right_hand, "Right")
        self.assertTrue(valid)
        # Wrist is the first 3 features (x, y, z)
        wrist_normalized = features[0:3]
        np.testing.assert_allclose(wrist_normalized, np.array([0.0, 0.0, 0.0]), atol=1e-6)

    def test_handedness_canonicalization(self):
        """8. Handedness behavior (mirroring X for Left hand)."""
        valid_r, feat_r, _ = self.normalizer.normalize(self.valid_right_hand, "Right")

        # Construct a left hand by mirroring the original right hand's X coordinate around the wrist
        left_hand = []
        wrist_x = self.valid_right_hand[0]['x']
        for lm in self.valid_right_hand:
            # mirror X relative to wrist
            dist_x = lm['x'] - wrist_x
            left_hand.append({'x': wrist_x - dist_x, 'y': lm['y'], 'z': lm['z']})

        valid_l, feat_l, _ = self.normalizer.normalize(left_hand, "Left")

        self.assertTrue(valid_r and valid_l)
        # After canonicalization, the features should be identical
        np.testing.assert_allclose(feat_r, feat_l, rtol=1e-5, atol=1e-5)

    def test_no_mutation(self):
        """9. Verify no accidental mutation of the original landmark array."""
        import copy
        original = copy.deepcopy(self.valid_right_hand)

        valid, _, _ = self.normalizer.normalize(self.valid_right_hand, "Right")
        self.assertTrue(valid)

        # Check that original is untouched
        self.assertEqual(original, self.valid_right_hand)

if __name__ == '__main__':
    unittest.main(verbosity=2)
