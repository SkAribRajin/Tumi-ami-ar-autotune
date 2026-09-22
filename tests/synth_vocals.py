"""
Synthetic vocal generators with KNOWN pitch contours, used by the quality
tests to measure in-tune error and vibrato retention objectively.

  ballad_vocal: sparse and dynamic. Notes sung ~30 cents flat, 5.5 Hz
                vibrato, a scoop into each note, soft/loud dynamics, rests
                with breath noise between phrases.
  rap_vocal:    dense and low. Already hard-quantised pitch (a previous
                autotune pass), fast syllable-rate amplitude chops,
                consonant noise bursts, no silence, heavy compression.
"""
import numpy as np

SR = 44100


def _render(f0_contour, amp_contour, sr=SR, num_harmonics=30, formants=(700, 1200, 2600)):
    """Additive synthesis of a vowel-like tone with a fixed formant envelope."""
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr
    out = np.zeros_like(f0_contour)
    for h in range(1, num_harmonics + 1):
        fh = f0_contour * h
        env = sum(1.0 / (1.0 + ((fh - fc) / (0.15 * fc)) ** 2) for fc in formants) + 0.05
        env = env * (fh < sr / 2 - 500)
        out += env * np.sin(h * phase) / h ** 0.5
    return out * amp_contour


def ballad_vocal(sr=SR, seed=0, notes_midi=(64, 67, 69, 67), note_s=1.2, rest_s=0.35,
                 detune_cents=-30.0, vibrato_hz=5.5, vibrato_cents=40.0):
    rng = np.random.default_rng(seed)
    segs_audio, segs_true = [], []
    for k, m in enumerate(notes_midi):
        n = int(note_s * sr)
        t = np.arange(n) / sr
        scoop = -80.0 * np.exp(-t / 0.06)                   # scoop up into the note
        vib = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t) * np.clip((t - 0.25) / 0.3, 0, 1)
        cents = detune_cents + scoop + vib
        f0 = 440.0 * 2 ** ((m - 69 + cents / 100.0) / 12.0)
        amp = (0.3 + 0.25 * (k % 2)) * np.clip(t / 0.05, 0, 1) * np.clip((note_s - t) / 0.1, 0, 1)
        segs_audio.append(_render(f0, amp, sr))
        segs_true.append(f0)
        r = int(rest_s * sr)
        segs_audio.append(0.003 * rng.standard_normal(r))    # breath / room tone
        segs_true.append(np.zeros(r))
    audio = np.concatenate(segs_audio)
    audio *= 0.7 / np.max(np.abs(audio))  # dynamic ballad: peaks well below full scale
    return audio.astype(np.float32), np.concatenate(segs_true)


def rap_vocal(sr=SR, seed=1, notes_midi=(45, 45, 48, 47, 45, 43, 45, 48), syllable_s=0.18,
              detune_cents=0.0):
    rng = np.random.default_rng(seed)
    n_syl = int(len(notes_midi) * 4)
    n = int(syllable_s * sr)
    f0_all, amp_all = [], []
    for s in range(n_syl):
        m = notes_midi[(s // 4) % len(notes_midi)]
        f0_all.append(np.full(n, 440.0 * 2 ** ((m - 69 + detune_cents / 100.0) / 12.0)))  # hard-quantised
        t = np.arange(n) / sr
        amp_all.append(0.6 + 0.4 * np.sin(np.pi * t / syllable_s))
    f0 = np.concatenate(f0_all)
    amp = np.concatenate(amp_all)
    voice = _render(f0, amp, sr, num_harmonics=60)
    # consonant bursts at syllable starts
    for s in range(n_syl):
        a = s * n
        voice[a:a + int(0.02 * sr)] += 0.3 * rng.standard_normal(int(0.02 * sr))
    voice = np.tanh(2.5 * voice / np.max(np.abs(voice)))     # heavy compression / saturation
    voice *= 0.9 / np.max(np.abs(voice))
    return voice.astype(np.float32), f0
