# Signal routing: stage by stage

What happens to the audio at each step of `run_pipeline`
(`src/autotune/pipeline.py`), the shape, dtype and scale to expect, and where
the old code (commit `9c5d530`) deviated from that.

Notation: `S` = number of input samples, `N` = frame size (2048),
`H` = hop (512), `F` = number of frames, `B` = N/2+1 = 1025 bins.
At 44.1 kHz: one frame = 46.4 ms, one hop = 11.6 ms, one bin = 21.5 Hz.

```
load ─► denoise? ─► (pre-emphasis? ─► detection only)
          │
          ├─► frame (windowed) ──────────────────────────────┐
          └─► frame (unwindowed) ─► YIN ─► cleanup ─► pitch, confidence
                                                  │
                                    quantise (hysteresis) ─► target
                                                  │
                       error × confidence ─► retune low-pass × strength ─► ratio
                                                  │
                    phase vocoder (+formants) ◄───┘
                                  │
                           overlap-add ─► underwater? ─► makeup gain ─► limiter ─► save
```

| # | Stage | Function | Output shape / dtype | Expected scale | Old deviation |
|---|---|---|---|---|---|
| 0 | Load | `io_utils.load_audio` | `(S,)` float32 | [−1, 1], mono, `config.sample_rate` | — (see note on stereo) |
| 1 | Noise reduction (optional, default on) | `filters.denoise_audio` | `(S,)` float32 | same level; only gap noise lowered | Edge samples ×10⁴ spike; voice subtracted on dense vocals (−7.8 dB); gate cut consistent-level material; tail zero-filled |
| 2 | Pre-emphasis (optional, **detection only**, default now off) | `filters.apply_filter` | `(S,)` float32 | lows cut (−26 dB at DC, +5.8 dB at Nyquist) | Was on by default in the UI; measurably worse for detection (below) |
| 3a | Framing for synthesis | `framing.frame_signal(audio, cfg)` | `(F, N)` float32 | Hann-windowed once | Tail: last partial hop dropped |
| 3b | Framing for detection | `frame_signal(audio, cfg, apply_window=False)` | `(F, N)` float32 | raw waveform, same frame centres | Detection used the **windowed** frames, which biases autocorrelation |
| 4 | Pitch detection | `pitch_detection.detect_pitch_track` | pitch `(F,)` float64 Hz (0 = unvoiced), confidence `(F,)` in [0, 1] | cent-accurate; no octave jumps | Integer lag (17–31 cent steps), octave errors, noise called voiced |
| 5 | Scale quantisation | `scales.quantize_pitch_track` | `(F,)` float64 Hz | one stable note per sung note | Per-frame nearest note flip-flopped near midpoints |
| 6 | Shift ratios | `pitch_shift.compute_shift_ratios` | `(F,)` float32 | within 2^(±2·strength/12); ~1.0 for in-tune and unvoiced frames | Up to ±10 semitones from the Hz rolling mean over octave errors |
| 7 | Phase vocoder | `phase_vocoder.phase_vocoder_shift` | `(F, N)` float32 | energy preserved (within 0.2–0.6 dB at ±3–6%); **identity at ratio 1.0** | Phase scrambled even at ratio 1.0 (SNR −2 to −3 dB); max-merge energy loss |
| 7b | Formant preservation | inside `phase_vocoder_shift(preserve_formants=True)` | same | gain limited to ±12 dB per bin | Separate `formant_preserve`: +4 to +22 dB per frame swings |
| 8 | Overlap-add | `framing.overlap_add(..., length=S)` | `(S,)` float32 | analysis w × synthesis w / Σw² = 1 | Length up to H−1 short |
| 9 | Underwater (optional) | `filters.apply_filter` + Butterworth LP | `(S,)` float32 | intentionally darker | — |
| 10 | Level | makeup gain (bounded 0.5–2.0, reported as `makeup_gain`) + `filters.soft_limit` | `(S,)` float32 | gain ≈ 1.00–1.05; peaks ≤ 0.98 | Unbounded RMS match to the denoised signal, then **whole-file** scaling by the single highest peak |
| 11 | Save | `io_utils.save_audio` → `soundfile.write` | 16-bit PCM WAV | anything outside ±1 would clip; the limiter guarantees ≤ 0.98 | Relied on the global clamp |

## Stage notes

### 0. Load
`soundfile.read` returns float64. Stereo is averaged to mono and cast to
float32. `resample_poly` is only used when the file's rate differs from 44.1
kHz. Its output is float64 and is cast back.

One remaining caveat (unchanged): averaging L and R is fine for a vocal
panned centre (identical channels, no level change). If the channels are
uncorrelated (wide stereo doubles or reverb) the mono sum is up to 3 dB
quieter than either channel.

### 1. Noise reduction
Noise is estimated only from genuine gaps: frames within 3 dB of the quietest
5%, and only when those are at least 10 dB below the loud frames. With no
such gaps (a dense rap verse) or a very clean stem (gaps more than 60 dB
down), the input is returned **unchanged**. The gain is a Wiener-style
`sqrt(max(1 − 1.5·N/|X|², 0.1²))`, smoothed over 3 frames. The signal is
padded before framing so edge samples are never divided by a near-zero window
sum.

Measured: dense rap 0.00 dB; ballad plus white noise, level −0.01 dB with
gap noise reduced by 5.7 dB; real uploads −0.01 dB (SNR vs input ≈ 35 dB,
i.e. only the noise changed).

### 2. Pre-emphasis
It only ever affects the detection copy, never the audio you hear. It is now
off by default in the pipeline, the UI and `app.py`. The comparison below was
measured with YIN on the same files:

| File | Voiced fraction (off → on) | Jumps > 0.3 st (off → on) |
|---|---|---|
| `upload_06cb36de.wav` | 0.74 → 0.71 | 6.1% → 10.5% |
| `upload_6d0e61b2.wav` | 0.59 → 0.47 | 7.9% → 7.5% |
| `upload_1acafa90.wav` | 0.76 → 0.61 | 13.3% → 13.1% |

Pre-emphasis boosts upper harmonics relative to the fundamental, which is
what makes a period detector lock onto a harmonic. The checkbox is still
there if you want to compare.

### 3. Framing
Both framings use identical frame starts (`i·H` into a signal padded with N
zeros in front and N+H behind), so detection frame `i` and synthesis frame
`i` cover the same audio. The extra H of tail padding ensures the last
samples are covered. `overlap_add(..., length=S)` trims the result to
exactly S.

### 4. Pitch detection
YIN (cumulative-mean-normalised difference) over lags for 60–1100 Hz, with
parabolic interpolation for sub-sample lags. It returns the aperiodicity
d′(τ) (0 = periodic, 1 = noise), which is converted to a soft confidence
(1 below 0.2, 0 above 0.45, linear between) and gated by loudness relative to
the track's own 95th-percentile frame level (−50 dB). Cleanup: octave
folding against a ±12-frame median, a 3-point median, and filling of gaps of
up to 2 frames.

### 5. Quantisation
A long (350 ms) running median gives the note centre, and ±0.25 semitone
hysteresis stops it flipping. A short (58 ms) median held for 58 ms more than
0.85 semitones from the current note forces an immediate, backfilled switch,
so fast notes aren't swallowed. See `retune-and-naturalness.md`.

### 6. Ratios
`error = 12·log2(target/detected)` in semitones, clamped to ±2, multiplied
by confidence, low-passed with time constant `retune_ms`, scaled by
`strength`, then `ratio = 2^(correction/12)`.

### 7. Phase vocoder
All maths runs in float64 and complex128, one frame at a time (memory
O(N), not O(F·N)). Frames arrive **already windowed** from stage 3a; the
vocoder doesn't window again. Per frame:

1. rotate by N/2 → rFFT (zero-phase: bin phase = phase at the frame centre)
2. peaks = every local maximum above −100 dB relative; each bin belongs to
   its nearest peak
3. per peak: true frequency from the phase advance since the last frame;
   integer bin offset `dk = round((ratio − 1)·f_bins)`; phase rotation
   `θ ← θ_prev + (ratio − 1)·Ω`, tracked per input partial
4. move each region by `dk`, multiply it by `e^{jθ}`, and **sum** collisions
5. optional formant correction: `Y *= env(k)/env(k/ratio)`, cepstral
   envelope with 30 coefficients, clipped to ±12 dB
6. irFFT, rotate back by N/2

At ratio 1.0, `dk = 0` and `θ = 0`, so `Y = X` exactly.

### 8. Overlap-add
`Σ frameᵢ·w / Σ w²`. With analysis and synthesis windows equal, this is the
standard weighted overlap-add and reconstructs exactly: frame → OLA identity
SNR 144.5 dB. The window is applied once in framing and once here, which is
correct.

### 10. Level
The phase vocoder loses 0.2–0.6 dB at typical correction ratios, from
partial cancellation where shifted regions overlap. The makeup gain restores
that and is reported as `result["makeup_gain"]`: 1.00–1.05 on every input
tested. It is bounded to ±6 dB so it can't hide a broken stage. The
limiter's gain curve is a minimum filter (±2.5 ms) followed by a smoothing
kernel no wider than that filter, which guarantees |output| ≤ 0.98 while
leaving samples more than ~5 ms from an over untouched (tested).
