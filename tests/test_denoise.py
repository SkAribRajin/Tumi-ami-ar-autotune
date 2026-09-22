import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autotune.filters import denoise_audio

class TestNoiseCancellation(unittest.TestCase):

    def setUp(self):
        self.sr = 44100
        self.duration = 2.0
        t = np.arange(int(self.sr * self.duration)) / float(self.sr)
        # Voice signal (first 1s voice + noise, second 1s pure noise)
        self.signal = np.zeros_like(t)
        self.signal[:self.sr] = 0.5 * np.sin(2 * np.pi * 300 * t[:self.sr])
        
        # Add background white noise floor
        rng = np.random.default_rng(42)
        self.noise = 0.08 * rng.standard_normal(len(t)).astype(np.float32)
        self.noisy_audio = self.signal + self.noise

    def test_output_shape_matches_input(self):
        cleaned = denoise_audio(self.noisy_audio, self.sr)
        self.assertEqual(len(cleaned), len(self.noisy_audio))

    def test_no_nan_or_inf(self):
        cleaned = denoise_audio(self.noisy_audio, self.sr)
        self.assertFalse(np.isnan(cleaned).any())
        self.assertFalse(np.isinf(cleaned).any())

    def test_noise_floor_attenuation_in_silence(self):
        cleaned = denoise_audio(self.noisy_audio, self.sr)
        # Energy during silence period (second half) should be reduced significantly
        silence_noisy_rms = np.sqrt(np.mean(self.noisy_audio[self.sr:] ** 2))
        silence_clean_rms = np.sqrt(np.mean(cleaned[self.sr:] ** 2))
        self.assertLess(silence_clean_rms, silence_noisy_rms * 0.4)


if __name__ == '__main__':
    unittest.main()
