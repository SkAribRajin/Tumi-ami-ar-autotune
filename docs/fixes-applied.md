# Fixes applied

Changelog of code changes, file by file, relative to commit `9c5d530`. The
reasoning and measurements behind each change are in `diagnosis.md`.

Test status: `python -m unittest discover tests` → **34 tests, all passing**
(17 existing + 17 new).

---

## `src/autotune/phase_vocoder.py`

**Replaced** `compute_stft`, `compute_instantaneous_frequency`,
`remap_and_lock_spectrum`, `shift_and_resynthesize_phase` and the core of
`phase_vocoder_shift` with a Laroche–Dolson phase-locked pitch shifter.

| Change | Why |
|---|---|
| Zero-phase analysis (frame rotated by N/2 before the rFFT, rotated back after the irFFT) | Bin phase becomes the sinusoid's phase at the frame centre; all main-lobe bins of a peak share it, so region locking is stable |
| Phase tracked per **input partial** as a rotation `θ ← θ_prev + (ratio − 1)·Ω` instead of a running phase per **output bin** | The old accumulator went stale whenever a peak moved one bin. That scrambled phase (SNR −2 to −3 dB at ratio 1.0) and was the cloudy sound. Now ratio 1.0 gives θ = 0 and an exact identity (SNR 144.5 dB) |
| Colliding bins are **summed** (complex `bincount`), not max-merged | The old code discarded energy |
| `find_peaks`: every local maximum above −100 dB (was −26 dB) | Weak upper harmonics were shifted with a lower peak's offset, which made them inharmonic |
| Formant preservation moved inside `phase_vocoder_shift(preserve_formants=True)`: gain `env(k)/env(k/ratio)`, cepstral envelope with 30 coefficients, clipped to ±12 dB | The known ratio gives the warped envelope directly. The old `formant_preserve` estimated it from the gappy shifted spectrum and swung +4 to +22 dB per frame |
| Processes one frame at a time | Memory O(N) instead of O(F·N); 57 s of audio runs in ~0.4 s for the PV stage |
| `formant_preserve` kept (CONTRACTS signature) with a docstring marking it as legacy | Other code or teammates may call it; the pipeline no longer does |

New signature (backward compatible):
`phase_vocoder_shift(frames, shift_ratios, config, preserve_formants=False, formant_coeffs=30, max_formant_gain_db=12.0)`.

## `src/autotune/pitch_detection.py`

Added YIN. The old `autocorrelate` and `detect_pitch_autocorrelation` are
kept unchanged (the root-level `test_phase_vocoder.py` imports
`autocorrelate`).

| Addition | Why |
|---|---|
| `_yin_cmndf`, `detect_pitch_yin` (vectorised, FFT-based, parabolic interpolation, first-dip rule plus "first dip within 0.1 of the best" fallback, digital silence reads as aperiodic) | Integer-lag argmax had 17–31 cent steps and octave errors |
| `voicing_confidence`: soft 0–1 weight from aperiodicity (0.2 → 0.45) with a loudness gate relative to the track's 95th percentile (−50 dB) | Replaces the hard `peak > 0.3` threshold that let noise through; works the same for quiet and mastered material |
| `clean_pitch_track`: octave folding against a ±12-frame median, 3-point median, bridging of gaps up to 2 frames | Octave jumps on real uploads 4.6–12.9% → 0–1.3% |
| `detect_pitch_track(audio, config)`: the pipeline's pitch stage (unwindowed frames → YIN → confidence → cleanup) | Detection previously ran on Hann-windowed frames |
| `detect_pitch_for_all_frames` now uses YIN internally; same signature and return value | CONTRACTS compatibility; all callers get the better detector |

## `src/autotune/pitch_shift.py`

| Change | Why |
|---|---|
| `compute_shift_ratios` rewritten as one confidence-weighted one-pole low-pass on the correction in semitones, time constant `retune_ms`, start state 0, clamp ±2 semitones | One code path covers robotic ↔ natural (see `retune-and-naturalness.md`). With `retune_ms=0` and no confidence, it equals the CONTRACTS formula `(target/detected)**strength` (tested) |
| Removed the vibrato rolling mean (`preserve_vibrato`) | It averaged octave errors in Hz and produced ±10 semitone shifts at strength 0.5. It also kept vibrato at retune 0, which made the robotic end unreachable |
| Removed the onset ramp (`onset_protection_ms`) | Now the filter's initial state; the old ramp was undone by the smoothing reset |
| Removed `smooth_shift_ratios` | Folded into `compute_shift_ratios`. It restarted every voiced run at the full, unsmoothed ratio, an instant snap on every syllable |
| `humanize_cents` kept | Unchanged behaviour, now applied in semitone space |

## `src/autotune/scales.py`

| Addition | Why |
|---|---|
| `quantize_pitch_track(pitches, scale, hop, sr, hysteresis=0.25, center_ms=350, jump_semitones=0.85, confirm_ms=58)` | Replaces per-frame `nearest_scale_note` in the pipeline. The 350 ms median centre plus hysteresis stops vibrato flip-flop; the 58 ms confirmed jump with backfill follows fast notes (≥ 70 ms) without lag. A running *mean* (first attempt) lagged fast steps by ~50 ms and produced −135 cent mis-corrections, so it's a median |
| `nearest_scale_note` unchanged | CONTRACTS compatibility |

## `src/autotune/filters.py`

| Change | Why |
|---|---|
| `denoise_audio` rewritten: pads the signal (no edge blow-up), estimates noise only from genuine gaps (frames within 3 dB of the quietest 5%, and at least 10 dB below the loud frames), returns the input untouched when there are no gaps or the stem is clean (gaps > 60 dB down), uses a Wiener gain with a 0.1 floor smoothed over 3 frames, removes the percentile gate | The old version subtracted voice on dense vocals (−7.8 dB), gated consistent-level material, and multiplied edge samples by up to 10⁴ (the spike behind the −33 dB drops) |
| New `soft_limit(audio, sr, ceiling=0.98, lookahead_ms=2.5)` | A local peak limiter (min-filter then smoothing kernel, provably ≤ ceiling) replaces whole-file scaling by the single highest peak |

## `src/autotune/pipeline.py`

| Change | Why |
|---|---|
| Uses `detect_pitch_track`, `quantize_pitch_track`, the new `compute_shift_ratios` (with `retune_ms` and `confidence`), and `phase_vocoder_shift(..., preserve_formants=...)` | New stages as above; the separate per-frame `formant_preserve` loop is gone |
| Level stage: RMS makeup gain **bounded to 0.5–2.0** and reported as `result["makeup_gain"]`, then `soft_limit` | The unbounded RMS match followed by `x *= 0.95/max(|x|)` turned one spike into a −7 to −33 dB loss for the whole file |
| `overlap_add(..., length=len(raw_audio))` | Output length now equals input length |
| `use_preemphasis` default **False** | Measured: pre-emphasis lowered the voiced fraction on every real upload and raised jumps on one |
| Removed the `preserve_vibrato` and `onset_protection_ms` arguments | Now properties of the retune filter. No caller in the repo passed them |
| Result dict adds `confidence` and `makeup_gain` | Diagnostics and plotting |

## `src/autotune/framing.py`

| Change | Why |
|---|---|
| `frame_signal(audio, config, apply_window=True)`: `False` returns unwindowed frames at the same positions | YIN needs the raw waveform |
| Tail padding N → N + H | The last partial hop (up to 11.6 ms) was dropped |
| `overlap_add(frames, config, pad_len, length=None)`: trims to exactly `length` when given | Output length = input length |

## `src/autotune/config.py`

Added `pitch_fmin = 60.0`, `pitch_fmax = 1100.0` and `note_hysteresis = 0.25`,
and rewrote the `retune_ms` comment to describe the robotic ↔ natural ranges.
No existing parameter changed.

## `app.py`, `templates/index.html`

- Retune slider range 0–150 → **0–200 ms**, labelled "Retune speed (robotic ← → natural)" with a tooltip.
- Pre-emphasis checkbox unchecked by default; `app.py` falls back to `false` if the field is missing.

## Tests

- **New** `tests/synth_vocals.py`: ballad and rap generators with known pitch
  contours (the rap generator can be detuned).
- **New** `tests/test_pipeline_quality.py`, 17 tests: PV identity and level
  preservation, pitch accuracy of the shifted output, YIN accuracy under
  3 cents from 73 to 880 Hz, silence handling, no octave jumps on dense low
  vocals, no flip-flop at a midpoint, fast-note switching, monotonic vibrato
  retention across `retune_ms`, snap at retune 0, end-to-end level within
  1 dB, bounded ratios, CONTRACTS formula at retune 0, denoiser behaviour on
  dense vs gapped material, limiter touching only overs, and level of a loud
  compressed vocal.
- The 17 existing tests (`test_phase_vocoder`, `test_denoise`,
  `test_underwater`) pass unchanged.

## Not changed (deliberately)

- Frame size 2048 / hop 512 (see `genre-robustness.md`).
- `naive_pitch_shift` (kept as the aliasing demo baseline).
- `underwater.py`, `visualization.py`, the demo scripts.
- Stereo-to-mono averaging in `load_audio`. It can be up to 3 dB quieter for
  decorrelated stereo; see `signal-routing.md`.

## Known limitations

- The phase vocoder moves regions by whole bins (21.5 Hz). The sub-bin
  remainder is carried by the phase progression, which leaves a small
  within-frame mismatch. The measured cost is 0.2–0.6 dB of level at ±3–6%
  shifts, which the makeup gain restores (1.00–1.05).
- A ±50-cent vibrato centred exactly between two notes can still switch
  target, because it is genuinely ambiguous.
- Everything above was verified with objective measurements on synthetic
  vocals and the uploads in `data/raw/`. No commercial ballad or rap stems
  were available, and none of the output has been listened to yet.
