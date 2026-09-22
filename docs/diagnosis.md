# Diagnosis: volume drop, cloudy sound, robotic output

Root-cause analysis of the three reported symptoms. Line numbers refer to the
code **before** the fix (commit `9c5d530`). All numbers were measured by
running that commit and the fixed code on the same inputs: three real uploads
from `data/raw/` and two synthetic vocals with known pitch (`tests/synth_vocals.py`).

## Summary

| Symptom | Root cause (old code) | Measured effect |
|---|---|---|
| Volume drop | A single spike (from the denoiser's edge normalisation or from a wrong shift ratio) feeds an RMS match and then a **whole-file** peak clamp | −7 to −14 dB on real uploads, −33.5 dB on a dense rap line |
| Volume drop | The denoiser treats the quietest 15% of frames as noise even when they are voice; its gate cuts consistent-level material | −7.8 dB on dense vocals before the clamp |
| Cloudy / smeared | The phase vocoder keeps phase per *output bin*, so when a peak moves one bin the phase is stale → at ratio 1.0 the output is no longer the input | SNR vs input at ratio 1.0: **−2 to −3 dB** (should be ≫ 60 dB) |
| Cloudy / smeared | Colliding bins merged with `max()`, harmonics below −26 dB ignored and shifted with the wrong offset | −0.1 to −1.1 dB of energy lost plus inharmonic partials |
| Robotic at moderate strength | Octave errors inside the vibrato "rolling mean" produce shifts of **±5–10 semitones** at strength 0.5 | Shift ratios 0.56 to 1.30 (up to 1043 cents) at strength 0.5 |
| Robotic at moderate strength | Integer-lag pitch detection jitters by 17–31 cents a frame; the target note flip-flops near a midpoint | 18–29% of voiced frames jump by more than 0.3 semitones; 4.6–12.9% jump by more than 6 semitones |

Before/after on the same files (strength 0.5, retune 40 ms, C major):

| Input | Old level Δ | New level Δ | Old max shift | New max shift |
|---|---|---|---|---|
| synthetic ballad (sparse, vibrato, 30 cents flat) | −0.02 dB | −0.00 dB | 360 c | 31 c |
| synthetic rap (dense, compressed, 35 cents sharp) | **−33.46 dB** | −0.04 dB | 258 c | 18 c |
| synthetic rap (already autotuned, in key) | **−33.90 dB** | −0.00 dB | 17 c | 6 c |
| real `upload_06cb36de.wav` | −9.32 dB | −0.01 dB | 954 c | 59 c |
| real `upload_6d0e61b2.wav` | −14.38 dB | −0.01 dB | 1043 c | 74 c |
| real `upload_1acafa90.wav` | −7.02 dB | −0.01 dB | 869 c | 77 c |

---

## 1. Volume drop

### 1a. Whole-file peak clamp after RMS match — `pipeline.py:101-110`

```python
gain = rms_raw / rms_corr
corrected_audio = corrected_audio * gain
max_val = np.max(np.abs(corrected_audio))
if max_val > 0.95:
    corrected_audio = corrected_audio * (0.95 / max_val)
```

This step turns one bad sample into a quieter song. Any single spike anywhere
in the output sets `max_val`, and the **entire file** is scaled down by it.

Trace of `upload_6d0e61b2.wav` (57 s): before this step the output peaked at
**26.4**, a crest factor of 40.5 dB against the input's 18.4 dB. The RMS match
left that spike in place and the clamp then scaled the whole file by about
1/28, which is **−25.7 dB**.

Two independent bugs create the spikes:

- **Denoiser edge blow-up, `filters.py:71-119`.** `denoise_audio` doesn't pad
  the signal, so the first and last few samples are covered only by the tail
  of a single Hann window. Normalisation divides by `win_sum`, which is about
  1e-8 there. Once spectral subtraction has changed the frame, the frame is no
  longer proportional to the window, and the division multiplies those samples
  by up to ~10⁴. On the dense rap line (sound from sample 0) the denoised
  output peaks at **37.4 at sample 5**, while the rest of the file peaks at
  0.35. Files that start in silence avoid this, which is why only some inputs
  lost 30 dB or more.
- **Wrong shift ratios, `pitch_shift.py:114-134`, amplified by
  `formant_preserve` (`phase_vocoder.py:302-308`).** See §3a for the ratio
  bug. `formant_preserve` estimates the "shifted" envelope from the shifted
  frame itself. That frame has zero-magnitude gaps, and `log(1e-8)` in those
  gaps drags the envelope down, so dividing by it boosts the frame. Measured
  per-frame gain reached **+22 dB** at ratio 1.0 and **+65 dB** at ratio 0.98
  on the rap line.

### 1b. The RMS match target is already too quiet — `pipeline.py:42-43, 102`

`raw_audio` is reassigned to the **denoised** signal on line 43, and line 102
matches against that. Whatever the denoiser removed stays removed, so a
denoiser that eats voice (§1c) produces a permanent level drop.

### 1c. Denoiser subtracts voice from dense vocals — `filters.py:85-104`

- Lines 85-88: the noise spectrum is the mean of the quietest 15% of frames,
  **whether or not those frames are quiet**. In a dense rap verse or a belted
  chorus they are still voice. Line 93 then subtracts 1.8× that voice
  spectrum from every frame.
- Lines 98-104: the gate threshold is 1.5 × the 20th-percentile frame level.
  On consistent-level (compressed) material most frames sit within 2× of the
  20th percentile, so most of the file is attenuated toward the 0.15 floor.

Measured on the dense synthetic rap line: **−7.8 dB** from the denoiser alone.
The phase vocoder lost another 4.5 dB (§2) before the clamp took 25.7 dB.

### 1d. Tail truncation — `framing.py:40`, `filters.py:122-123`

`frame_signal` pads the end with only N zeros, so the last partial hop (up to
511 samples, 11.6 ms) was never covered by a frame. The output was up to one
hop shorter than the input: 448 samples short on `upload_6d0e61b2.wav`. The
denoiser zero-filled its own uncovered tail in the same way.

---

## 2. Cloudy / smeared / distorted sound

### 2a. Phase vocoder is not coherent — `phase_vocoder.py:119-172` (`remap_and_lock_spectrum`)

The test that exposes it: run the phase vocoder at a constant ratio of **1.0**.
A correct implementation returns the input (limited only by float precision).

| File | Old SNR vs input at ratio 1.0 | New |
|---|---|---|
| `my_voice.wav` | −2.1 dB | 144.5 dB |
| `upload_06cb36de.wav` | −3.1 dB | 144.5 dB |
| `upload_6d0e61b2.wav` | −3.4 dB | 144.5 dB |
| `upload_1acafa90.wav` | −3.4 dB | 144.5 dB |

An SNR below 0 dB means the output waveform is less correlated with the input
than silence would be. The level is roughly right, but the phases are
scrambled, and scrambled phase is what "cloudy" sounds like. Because this
happens at ratio 1.0, it affects **every** frame, including unvoiced frames
and frames that need no correction. That is why the sound was bad even at
moderate strength.

Causes, by line:

1. **Line 145** `target_bin = int(round(target_freq / bin_spacing))`: even at
   ratio 1.0 the peak's measured frequency rarely rounds back to its own bin.
   Peaks move ±1 bin from frame to frame with vibrato and measurement noise.
2. **Lines 148-153** `running_phase[target_bin] += phase_advance`: the
   accumulator is indexed by **output bin**. It only advances on frames where
   a peak lands in that bin. When a partial moves from bin k to k+1, it picks
   up `running_phase[k+1]`, which was last advanced some unknown number of
   hops ago. The result is a random phase jump per partial per frame, so
   overlapping frames interfere destructively.
3. **Lines 151-153, 168-170** `if magnitude[p] > new_magnitude[target_bin]`:
   when two regions land on the same bin, the louder one wins and the other's
   energy is discarded. That's a small level loss (§1) and holes in the
   spectrum.
4. **Line 93** `min_rel_height=0.05`: only peaks within −26 dB of the
   loudest bin count. Upper vocal harmonics are often −30 to −50 dB, so their
   bins fall into the region of a lower peak (line 113 splits at the midpoint
   between the remaining peaks) and are shifted by **that** peak's bin offset.
   The shift needed grows with frequency, so they end up at the wrong
   frequency, which makes the tone inharmonic and metallic.
5. **Lines 49-58** `compute_stft`: the frame isn't rotated to zero phase, so
   bin phases include the window's linear-phase term. That makes the
   peak-relative offsets of lines 166-167 less stable than they need to be.

### 2b. Formant stage adds frame-to-frame gain modulation — `phase_vocoder.py:277-315`

See §1a: the per-frame gain varies by up to +22 dB depending on how sparse the
shifted spectrum is. Frame-to-frame loudness changes at the hop rate (86 Hz)
are heard as roughness, not as volume.

### 2c. What is *not* broken

- `framing.overlap_add` divides by Σw² with the same Hann window used for
  analysis. That is correct weighted overlap-add (frame+OLA identity SNR
  144.5 dB). The window is applied twice on purpose (analysis + synthesis)
  and normalised correctly, **not double-applied by mistake**.
- Frame size 2048 / hop 512 at 44.1 kHz is adequate (see `genre-robustness.md`).

---

## 3. Robotic even at moderate strength

### 3a. Vibrato "rolling mean" averages octave errors — `pitch_shift.py:114-134`

```python
rolling_mean[i] = np.mean(detected_pitches[start_i:end_i][v_in_win])   # in Hz
effective_target[i] = target_pitches[i] * (detected_pitches[i] / rolling_mean[i])
```

The detector made octave errors (§3b). One frame detected at 760 Hz in a
window of 380 Hz frames gives a rolling mean near 420 Hz and a "vibrato ratio"
of 1.8. The target is then multiplied by that. At strength 0.5 this produced
ratios of **0.56 to 1.30** (about ±5 to 10 semitones), where the largest
legitimate correction is ±0.5 semitones. Those frames are both the loudest
artefacts and the source of the spikes in §1a. The mean is also taken in Hz
across note boundaries, so at every legato note change the correction is
computed against a blend of two notes.

It also defeated the "robotic" end of the control: with
`preserve_vibrato=True` (the default, not exposed in the UI) the output pitch
is `target × detected / rolling_mean`, which keeps vibrato **even at
retune 0**. The pipeline could not produce the hard-tuned sound on purpose,
only by accident through artefacts.

### 3b. Pitch detector — `pitch_detection.py:97-133`

- **Line 127** `peak_lag = np.argmax(search_region) + min_lag`: integer lags
  only. At 440 Hz one lag step is 17 cents; at 800 Hz it is 31 cents. The
  detected pitch hops between neighbouring lags, and that jitter becomes
  frame-rate pitch flutter in the output.
- **Line 99** `corr / corr[0]` on a **Hann-windowed** frame (the pipeline
  passes windowed frames, `pipeline.py:57-59`): the window tapers the
  autocorrelation toward long lags, which biases the argmax and causes
  octave errors.
- **Line 130** `confidence_threshold = 0.3`: noise and breaths pass. On
  `my_voice.wav` (a −48 dBFS recording of room noise with no clear singing)
  the old detector called 65% of frames voiced, with 13% of consecutive frames
  jumping by an octave or more. The new detector calls 11% voiced.

Measured frame-to-frame jumps on real uploads (old → new): > 0.3 semitones
17.5–29.4% → 6.1–14.3%; > 6 semitones 4.6–12.9% → 0–1.3%. What remains is
mostly real note changes.

### 3c. Target note flip-flop — `pipeline.py:66-69`

`nearest_scale_note` is called independently per frame. A singer holding a
note near the midpoint between two scale notes, or with vibrato that crosses
the midpoint, makes the target alternate between the two notes. That is a
±1–2 semitone square wave at the vibrato rate, which is robotic at any
strength.

### 3d. Smoothing restarts with an instant snap — `pitch_shift.py:210-218`

`smooth_shift_ratios` resets at every unvoiced frame and starts each voiced
run at the **full, unsmoothed** ratio (`smoothed_log[i] = log_ratios[i]`).
Every syllable onset is therefore hard-snapped whatever `retune_ms` is set to,
and rap has an onset every 100–200 ms. The separate onset ramp
(`pitch_shift.py:137-151`) acts on the strength *before* this reset, so the
two mechanisms work against each other.

---

## Verification

`tests/test_pipeline_quality.py` pins each finding (17 tests):
PV identity at ratio 1.0 (SNR > 60 dB), level within 1 dB at ratios 0.94–1.06
with and without formants, YIN accuracy under 3 cents from 73 to 880 Hz,
silence not reported as periodic, no octave jumps on a dense low vocal, no
flip-flop at a note midpoint, fast notes switching without lag, monotonic
vibrato retention across `retune_ms`, level preserved end to end, the
denoiser leaving dense vocals untouched, and the limiter touching only the
overs. The whole suite (34 tests) passes.
