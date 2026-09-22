"""
Regression tests for the bugs documented in docs/diagnosis.md. Each test
pins down a measured failure of the old code:

  - phase vocoder was not an identity at ratio 1.0 (SNR -2..-3 dB)   -> cloudy
  - output level dropped (up to -25 dB) via spikes + whole-file clamp -> volume
  - integer-lag autocorrelation jitter / octave errors               -> warble
  - per-frame note choice flip-flopped near a note midpoint          -> robotic
  - denoiser subtracted voice from dense vocals (-9 dB)              -> volume
  - retune_ms must move output continuously natural <-> robotic
"""
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from autotune.config import AutoTuneConfig
from autotune.framing import frame_signal, overlap_add
from autotune.phase_vocoder import phase_vocoder_shift
from autotune.pitch_detection import detect_pitch_yin, detect_pitch_track
from autotune.scales import build_scale_midi_set, quantize_pitch_track
from autotune.pitch_shift import compute_shift_ratios
from autotune.filters import denoise_audio, soft_limit
from autotune.io_utils import save_audio
from autotune.pipeline import run_pipeline
from synth_vocals import ballad_vocal, rap_vocal, SR


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def snr_db(ref, x):
    ref = np.asarray(ref, dtype=np.float64)
    err = ref - np.asarray(x, dtype=np.float64)
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum(err ** 2) + 1e-20))


def harmonic_tone(f0, seconds=1.0, harmonics=15):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, harmonics + 1))
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


class TestPhaseVocoderFidelity(unittest.TestCase):

    def setUp(self):
        self.cfg = AutoTuneConfig()

    def test_ratio_one_is_identity(self):
        x, _ = ballad_vocal()
        frames, pad = frame_signal(x, self.cfg)
        y = overlap_add(phase_vocoder_shift(frames, np.ones(len(frames)), self.cfg), self.cfg, pad, len(x))
        self.assertEqual(len(y), len(x))
        self.assertGreater(snr_db(x, y), 60.0)

    def test_small_shifts_preserve_level(self):
        x, _ = ballad_vocal()
        frames, pad = frame_signal(x, self.cfg)
        for ratio in (0.94, 0.97, 1.03, 1.06):
            for formants in (False, True):
                y = overlap_add(phase_vocoder_shift(frames, np.full(len(frames), ratio), self.cfg,
                                                    preserve_formants=formants), self.cfg, pad)
                self.assertLess(abs(rms_db(y) - rms_db(x)), 1.0, (ratio, formants))

    def test_output_pitch_matches_ratio(self):
        x = harmonic_tone(200.0)
        frames, pad = frame_signal(x, self.cfg)
        y = overlap_add(phase_vocoder_shift(frames, np.full(len(frames), 2 ** (1 / 12)), self.cfg),
                        self.cfg, pad)
        p, _ = detect_pitch_track(y, self.cfg)
        cents = 1200 * np.log2(np.median(p[p > 0]) / (200.0 * 2 ** (1 / 12)))
        self.assertLess(abs(cents), 5.0)


class TestPitchDetection(unittest.TestCase):

    def test_yin_is_cent_accurate(self):
        cfg = AutoTuneConfig()
        for f0 in (73.4, 146.8, 311.1, 523.3, 880.0):
            frames, _ = frame_signal(harmonic_tone(f0, 0.5), cfg, apply_window=False)
            p, ap = detect_pitch_yin(frames[4:-4], SR)
            err = np.abs(1200 * np.log2(p / f0))
            self.assertLess(np.median(err), 3.0, f0)
            self.assertLess(np.median(ap), 0.1, f0)

    def test_silence_is_not_periodic(self):
        _, ap = detect_pitch_yin(np.zeros((3, 2048)), SR)
        self.assertTrue(np.all(ap == 1.0))

    def test_no_octave_jumps_on_dense_low_vocal(self):
        cfg = AutoTuneConfig()
        x, _ = rap_vocal()
        p, conf = detect_pitch_track(x, cfg)
        v = conf > 0
        semis = 12 * np.log2(np.where(v, p, 1.0))
        jumps = np.abs(np.diff(semis))[v[1:] & v[:-1]]
        self.assertGreater(v.mean(), 0.8)
        self.assertEqual(int(np.sum(jumps > 6)), 0)


class TestNoteQuantisation(unittest.TestCase):

    def test_hysteresis_stops_flip_flop(self):
        # hovering at the C/C# midpoint with +/-30 cent vibrato
        n = 300
        cents = 50 + 30 * np.sin(2 * np.pi * 5.5 * np.arange(n) * 512 / SR)
        pitches = 261.63 * 2 ** (cents / 1200)
        targets = quantize_pitch_track(pitches, build_scale_midi_set("C", "chromatic"))
        self.assertLessEqual(len(np.unique(np.round(targets, 2))), 1)

    def test_fast_note_steps_switch_immediately(self):
        steps = np.repeat([110.0, 130.81, 123.47, 110.0], 15)  # A2 C3 B2 A2, 174 ms each
        targets = quantize_pitch_track(steps, build_scale_midi_set("C", "major"))
        np.testing.assert_allclose(targets, steps, rtol=1e-3)


class TestRetuneContinuum(unittest.TestCase):
    """retune_ms alone must move the result continuously from robotic to natural."""

    @classmethod
    def setUpClass(cls):
        x, _ = ballad_vocal()
        cls.dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.dir, "ballad.wav")
        save_audio(cls.path, x, SR)
        cls.input_level = rms_db(x)
        cls.results = {}
        for rt in (0, 25, 100):
            cfg = AutoTuneConfig(scale_root="C", scale_type="chromatic", retune_ms=rt)
            cls.results[rt] = run_pipeline(cls.path, cfg)

    def _vibrato_cents(self, audio):
        from scipy.signal import butter, filtfilt
        cfg = AutoTuneConfig()
        p, conf = detect_pitch_track(audio, cfg)
        v = conf > 0.8
        cents = 1200 * np.log2(np.where(v, p, 440.0) / 440.0)
        b, a = butter(2, [4 / 43.07, 8 / 43.07], btype='band')
        runs, i = [], 0
        while i < len(v):
            if not v[i]:
                i += 1
                continue
            j = i
            while j < len(v) and v[j]:
                j += 1
            if j - i > 30:
                runs.append(filtfilt(b, a, cents[i:j])[8:-8])
            i = j
        return np.sqrt(np.mean(np.concatenate(runs) ** 2))

    def test_vibrato_retention_increases_with_retune(self):
        vib = [self._vibrato_cents(self.results[rt]["corrected_audio"]) for rt in (0, 25, 100)]
        self.assertLess(vib[0], vib[1])
        self.assertLess(vib[1], vib[2])
        self.assertLess(vib[0], 6.0)    # robotic: vibrato flattened
        self.assertGreater(vib[2], 17.0)  # natural: most of the ~21 cent vibrato kept

    def test_retune_zero_snaps_to_note(self):
        cfg = AutoTuneConfig()
        p, conf = detect_pitch_track(self.results[0]["corrected_audio"], cfg)
        v = conf > 0.8
        cents = 1200 * np.log2(p[v] / 440.0)
        off = np.abs(cents - 100 * np.round(cents / 100))
        self.assertLess(np.median(off), 5.0)

    def test_level_is_preserved(self):
        for rt, r in self.results.items():
            self.assertLess(abs(rms_db(r["corrected_audio"]) - self.input_level), 1.0, rt)

    def test_ratios_bounded(self):
        for r in self.results.values():
            self.assertLess(np.max(np.abs(12 * np.log2(r["shift_ratios"]))), 2.0 + 1e-6)

    def test_retune_zero_matches_contract_formula(self):
        det = np.array([0.0, 430.0, 450.0, 0.0])
        tgt = np.array([0.0, 440.0, 440.0, 0.0])
        r = compute_shift_ratios(det, tgt, strength=0.5)
        expected = np.array([1.0, (440 / 430) ** 0.5, (440 / 450) ** 0.5, 1.0])
        np.testing.assert_allclose(r, expected, rtol=1e-5)


class TestLevelAndDynamics(unittest.TestCase):

    def test_dense_vocal_not_attenuated_by_denoise(self):
        x, _ = rap_vocal()
        self.assertLess(abs(rms_db(denoise_audio(x, SR)) - rms_db(x)), 0.5)

    def test_denoise_reduces_noise_in_gaps(self):
        x, _ = ballad_vocal()
        noisy = (x + 0.01 * np.random.default_rng(0).standard_normal(len(x))).astype(np.float32)
        cleaned = denoise_audio(noisy, SR)
        self.assertLess(rms_db(cleaned - x), rms_db(noisy - x) - 3.0)
        self.assertLess(abs(rms_db(cleaned) - rms_db(noisy)), 0.5)

    def test_limiter_only_touches_overs(self):
        x = 0.5 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
        x[SR // 2] = 3.0  # one spike
        y = soft_limit(x.astype(np.float32), SR)
        self.assertLessEqual(np.max(np.abs(y)), 0.98 + 1e-6)
        far = np.r_[0:SR // 2 - 500, SR // 2 + 500:SR]
        np.testing.assert_allclose(y[far], x[far], atol=1e-6)

    def test_loud_compressed_vocal_keeps_its_level(self):
        x, _ = rap_vocal(detune_cents=35.0)  # peaks at 0.9, needs real correction
        path = os.path.join(tempfile.mkdtemp(), "rap.wav")
        save_audio(path, x, SR)
        r = run_pipeline(path, AutoTuneConfig(scale_root="C", scale_type="major", retune_ms=10))
        self.assertLess(abs(rms_db(r["corrected_audio"]) - rms_db(x)), 1.0)
        self.assertLessEqual(np.max(np.abs(r["corrected_audio"])), 0.98 + 1e-6)


if __name__ == '__main__':
    unittest.main()
