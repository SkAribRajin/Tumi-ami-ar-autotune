# Genre robustness: sparse ballads and dense, heavily processed rap

The goal is one algorithm with one set of defaults that handles both of these
without per-genre modes:

- **Sparse, dynamic ballad** (soft acoustic vocals in the style of Ed
  Sheeran's "Perfect"): quiet passages, long notes with vibrato, scoops, rests
  with breaths and room tone, wide dynamic range, mid-to-high register and
  falsetto.
- **Dense, heavily processed melodic rap** (Travis Scott-style heavy
  autotune): little or no silence, low register, fast syllables, consonant
  bursts, heavy compression and saturation, often *already* hard-tuned,
  frequently doubled or with ad-libs and reverb.

## Assumptions in the old code that held for only one of the two

| Stage | Old assumption | Breaks on | What happened |
|---|---|---|---|
| Denoiser noise estimate | The quietest 15% of frames are noise | Dense rap (no silence) | Subtracted 1.8× the voice from itself: −7.8 dB |
| Denoiser gate | Frames below 2× (1.5 × 20th percentile) are background | Compressed vocals (every frame is at a similar level) | Most frames pushed toward 0.15× |
| Denoiser edges | Files start and end in silence | Stems that start on a word (typical rap acapella) | Edge sample ×10⁴ → whole-file clamp: −33 dB total |
| Voicing | Autocorrelation peak > 0.3 means voiced | Ballad breaths and room tone; rap consonants | Noise pitch-shifted; `my_voice.wav` (room noise) called 65% voiced |
| Pitch resolution | Integer lag is precise enough | High ballad notes (17–31 cent steps above 440 Hz) | Frame-rate flutter |
| Octave errors | A windowed ACF argmax finds the fundamental | Low voices with strong 2nd harmonics, saturated rap | 4.6–12.9% octave jumps, then ±10 st shifts through the vibrato mean |
| Pitch range | 80–1000 Hz | Low rap and baritone registers below 80 Hz | Low notes unvoiced or octave-doubled |
| Note choice | Per-frame nearest note | Ballad vibrato near a midpoint | Target flip-flop warble |
| Smoothing reset | Unvoiced frames are rare | Rap (a consonant every 100–200 ms) | Instant snap at every syllable, whatever the retune setting |
| Pre-emphasis | Boosting highs helps detection | Low voices (fundamental weakened by up to 26 dB) | Fewer voiced frames, more pitch jumps (measured) |

## What changed, and why it works for both

### Frame size: unchanged, 2048 at 44.1 kHz (46 ms, 21.5 Hz bins), hop 512

No per-genre frame size is needed.

- **Low rap voices.** Harmonics are resolved (≥ 4 bins apart) from about
  86 Hz up. Below that, neighbouring harmonics share a region. For autotune
  ratios (≤ ±6%), neighbouring harmonics need bin offsets that differ by only
  `0.06 × f0 / 21.5 Hz ≈ 0.2` bins, so they round to the same integer shift
  and the shared region moves correctly. The phase vocoder now finds every
  local maximum (down to −100 dB) instead of only peaks within −26 dB, so
  weak upper harmonics get their own correct offset.
- **Fast rap syllables.** The 11.6 ms hop gives about 8–15 pitch estimates
  per 100–180 ms syllable. The 46 ms analysis window is shorter than a
  syllable.
- **YIN's integration window** is N − max_lag = 2048 − 735 = 1313 samples
  (30 ms): enough periods of a 60 Hz voice, short enough for note changes.
- A larger frame (4096) would help only below ~60 Hz, and it would double
  the smearing of fast notes. A smaller one (1024) would leave low voices
  unresolved.

### Pitch range: 60–1100 Hz for everyone (`config.pitch_fmin/fmax`)

This covers deep male rap and baritone registers through soprano and
falsetto. With YIN, a wide range costs little accuracy: YIN takes the
*first* dip below threshold, which stops long lags causing octave-low errors.
When nothing clears the threshold, it takes the first dip within 0.1 of the
deepest one, which stops the 2T dip winning on breathy or saturated frames.

### Confidence: relative, soft, and level-independent

- Aperiodicity (YIN d′) maps to a weight: 1 below 0.2, 0 above 0.45,
  linear between. Measured on the louder half of the frames, every input
  tested (both synthetic archetypes and the real uploads) has a median
  aperiodicity of 0.00–0.04 and a 90th percentile of at most 0.15, all in
  the full-weight zone, so saturation and compression don't cost correction.
  Consonants, breaths and room noise (> 0.45) get none.
- The loudness gate is relative: −50 dB below the track's own 95th-percentile
  frame level, not an absolute threshold. A −46 dBFS phone recording and a
  −8 dBFS mastered acapella are treated the same way.
- The weight scales the input to the retune filter, so a doubled, reverb-y or
  ad-lib-heavy frame (higher aperiodicity) is corrected less. That is the
  safe default when the detector is less sure.

### Octave-error cleanup and gap filling

Frames more than 7 semitones from the ±12-frame running median are folded
by octaves. A 3-point median then removes single-frame outliers. Dropouts of
up to 2 frames inside a note are bridged at half confidence, so a consonant
flam doesn't restart the correction. Measured on the dense low synthetic rap
line: **0** octave jumps (test `test_no_octave_jumps_on_dense_low_vocal`).
On real uploads: 0–1.3% of frames jump more than 6 semitones, versus
4.6–12.9% before.

### Denoiser: noise estimated only from real gaps

The denoiser estimates noise from frames within 3 dB of the quietest 5%,
provided those are at least 10 dB below the loud frames.

- Ballad with rests: gaps found, noise reduced (gap noise −5.7 dB), voice
  level unchanged (−0.01 dB).
- Dense rap: no gaps, so the audio is returned untouched (0.00 dB). With no
  gaps the voice masks stationary noise anyway.
- Clean studio stem (gaps more than 60 dB down): untouched.

### Already hard-tuned input

A Travis Scott-style vocal that is already tuned has `error ≈ 0` in every
voiced frame. The correction stays near 0 and the ratio near 1.0, and at
ratio 1.0 the phase vocoder is an exact identity. Measured: a synthetic
already-tuned rap line comes out with a level change of −0.09 to −0.13 dB
across all retune settings, and its pitch offset from the note stays at
0.0 cents.
The old code took −33.9 dB off the same input. If the chosen scale doesn't
match the track, the fast-note rule and hysteresis still stop the target
from flickering between two notes.

### Scale quantisation

The same settings serve both: a 350 ms median centre for vibrato (ballad)
and a 58 ms confirmed jump for fast notes (rap). The synthetic rap line
(notes 180 ms long, 35 cents sharp) is corrected to within 0.0–1.2 cents at
retune 0–200 ms. Notes down to 70 ms are followed without lag.

## Measured on both archetypes with the same defaults

C chromatic, strength 1.0 (`tests/synth_vocals.py`):

| retune_ms | Ballad: offset from note | Ballad: vibrato kept | Rap (35 c sharp): offset | Rap (already tuned): level Δ |
|---|---|---|---|---|
| 0 | 1.4 c | 24%* | 0.0 c | −0.09 dB |
| 25 | 9.8 c | 60% | 0.3 c | −0.11 dB |
| 50 | 16.8 c | 80% | 0.7 c | −0.12 dB |
| 100 | 24.1 c | 91% | 0.9 c | −0.13 dB |
| 200 | 28.7 c | 96% | 1.2 c | −0.13 dB |

The ballad's "offset from note" grows with retune because vibrato is kept.
A 40-cent vibrato has a median |offset| of ~28 cents even when perfectly
centred. \*Measurement floor; see `retune-and-naturalness.md`.

## What was not tested

- **No commercial stems.** The two archetypes are synthetic stand-ins with
  known pitch contours, plus the uploads already in `data/raw/`. Before
  relying on these defaults for release work, run the pipeline on real
  acapellas of both kinds and listen.
- **All results are objective measurements**: level, pitch accuracy,
  vibrato energy, and SNR at ratio 1. Nobody has listened to them yet.
- Polyphonic input (stacked harmonies in one file) is outside what a
  single-pitch corrector can do. YIN follows the dominant voice and the
  confidence weight drops, so the other voices are shifted along with it,
  only gently.
