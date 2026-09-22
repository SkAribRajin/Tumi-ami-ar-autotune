import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autotune.underwater import (
    apply_underwater_effect,
    apply_deep_underwater_filter,
    generate_speech_gated_bubbles,
)

class TestUnderwaterEffect(unittest.TestCase):

    def setUp(self):
        self.sr = 44100
        self.duration = 2.0
        t = np.arange(int(self.sr * self.duration)) / float(self.sr)
        self.audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    def test_output_shape_matches_input(self):
        processed = apply_underwater_effect(self.audio, self.sr)
        self.assertEqual(len(processed), len(self.audio))

    def test_no_nan_or_inf(self):
        processed = apply_underwater_effect(self.audio, self.sr)
        self.assertFalse(np.isnan(processed).any())
        self.assertFalse(np.isinf(processed).any())

    def test_filter_attenuates_high_frequencies(self):
        t = np.arange(self.sr) / float(self.sr)
        high_freq = (0.5 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
        filtered = apply_deep_underwater_filter(high_freq, self.sr, cutoff=480.0)
        self.assertLess(np.max(np.abs(filtered)), np.max(np.abs(high_freq)) * 0.1)

    def test_bubble_track_generation(self):
        rng = np.random.default_rng(42)
        bubbles = generate_speech_gated_bubbles(self.audio, self.sr, rng)
        self.assertEqual(len(bubbles), int(2.0 * self.sr))
        self.assertGreater(np.max(np.abs(bubbles)), 0.01)

    def test_no_clipping(self):
        processed = apply_underwater_effect(self.audio * 2.0, self.sr)
        self.assertLessEqual(np.max(np.abs(processed)), 0.96)


if __name__ == '__main__':
    unittest.main()
