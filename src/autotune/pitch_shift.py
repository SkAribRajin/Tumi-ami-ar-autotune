import numpy as np

"""
Naive Pitch Shifting via Resampling
=====================================

The idea: to change pitch, we change how "densely" we read the original
samples. If we read faster (skip ahead), the waveform compresses in time,
which raises pitch when played back at the normal rate. If we read slower
(re-visit nearby points via interpolation), the waveform stretches, which
lowers pitch.

We do this using simple linear interpolation (np.interp) - and deliberately
so, because linear interpolation does NOT apply a low-pass filter before
resampling. This is exactly what makes it "naive" and lets us demonstrate
a real course concept: ALIASING.

Reminder from the sampling theorem: before you reduce your effective sample
rate (skip over samples), you must remove frequency content above the new
Nyquist limit, or higher frequencies "fold back" and appear as false lower
frequencies - this is aliasing. Proper resamplers (like scipy's resample_poly,
which you already used in io_utils.py) apply this filter. Here, we skip that
filter on purpose so you can hear/see the difference against the phase
vocoder later.

How the shift actually works:

original indices:   0, 1, 2, 3, ..., frame_size-1
read positions:      0, shift_ratio, 2*shift_ratio, 3*shift_ratio, ...

Example: frame_size=8, shift_ratio=2.0 (shift up an octave)
read positions: 0, 2, 4, 6, 8, 10, 12, 14
Notice positions 8-14 don't exist in our original frame (max index is 7)!
np.interp will just clamp these to the last sample - this clamping is itself
one of the naive method's artifacts (not a bug, just a real limitation).

Example: frame_size=8, shift_ratio=0.5 (shift down an octave)
read positions: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
We only ever use the FIRST HALF of the frame, stretched to fill the whole
output - the back half of the original frame (indices 4-7) never gets used.
This is why naive pitch shifting also distorts information, not just pitch.
"""

def naive_pitch_shift(frame, shift_ratio):
    """
    frame: np.ndarray, shape (frame_size,) - one windowed audio frame
    shift_ratio: float - e.g. 1.05 = shift up ~1 semitone, 0.95 = shift down

    Returns: np.ndarray, shape (frame_size,) - shifted frame, same length,
    so it can drop straight into overlap_add() unchanged.
    """
    n = len(frame)
    original_indices = np.arange(n)
    read_positions = np.arange(n) * shift_ratio

    # np.interp reads frame[] at these possibly-fractional positions,
    # linearly blending between the two nearest real samples.
    # Positions beyond n-1 are clamped to the last sample automatically.
    shifted = np.interp(read_positions, original_indices, frame)

    return shifted.astype(np.float32)


"""
Computing Shift Ratios from Detected + Target Pitch
=====================================================

Once pitch_detection.py tells us the detected pitch per frame, and scales.py
tells us the nearest correct note (target pitch), we need a single number:
how much to shift each frame by.

The obvious approach: ratio = target_pitch / detected_pitch
But we also want a "correction strength" knob (0.0 = no correction at all,
1.0 = full correction) so we can demo both natural, subtle correction and
the classic hard robotic snap.

Why we use exponentiation instead of linear blending for partial strength:
Pitch is perceived LOGARITHMICALLY - this is the exact same reasoning
scales.py already uses (MIDI numbers are a log2-based scale, see freq_to_midi).
So "50% correction" should mean "halfway in semitone-distance", not "halfway
in raw Hz-ratio". This is done with:

    ratio = (target/detected) ** strength

When strength = 1.0: ratio = target/detected (full correction)
When strength = 0.0: ratio = 1.0 (no change at all, since anything^0 = 1)
When strength = 0.5: ratio = halfway between them ON THE LOG SCALE
"""

"""
One correction filter instead of separate modes
=================================================
Work in semitones. Per frame:
    error[i] = 12*log2(target/detected)        how far this frame is from its note
    u[i]     = confidence[i] * error[i]        confidence gating (0 for unvoiced)
    c[i]     = c[i-1] + alpha*(u[i] - c[i-1])  one-pole low-pass, time constant retune_ms
    ratio[i] = 2 ** (strength * c[i] / 12)

The output pitch is detected + c, and the pitch is detected = note + deviation.
So output = note + (deviation - LPF(deviation)) = note + HPF(deviation).
The retune time constant tau sets the crossover fc = 1/(2*pi*tau):
  - deviations SLOWER than fc (a note sung flat, drift) are removed -> in tune
  - deviations FASTER than fc (vibrato ~5-7 Hz, scoops, onsets) pass through
tau -> 0 pushes fc above the vibrato rate, which flattens vibrato and turns
note changes into steps: the robotic sound. Larger tau keeps them: natural.
Onset protection is the filter's starting state (c = 0 at a note start, so
the attack is heard as sung while the correction ramps in). Vibrato
preservation is its high-pass complement. No separate code paths.
See docs/retune-and-naturalness.md.
"""

def compute_shift_ratios(detected_pitches, target_pitches, strength=1.0,
                         retune_ms=0.0, confidence=None, hop_size=512, sample_rate=44100,
                         humanize_cents=0.0, max_correction_semitones=2.0):
    """
    detected_pitches: shape (num_frames,) - detected pitch in Hz, 0 = unvoiced
    target_pitches:   shape (num_frames,) - target scale note in Hz, 0 = unvoiced
    strength:         0.0 = no correction, 1.0 = full correction (scales the
                      correction in log/semitone space, same as before)
    retune_ms:        correction low-pass time constant (0 = per-frame snap,
                      identical to the original (target/detected)**strength)
    confidence:       optional 0..1 voicing weight per frame (defaults to 1
                      for voiced frames). Frames with low confidence get
                      proportionally less correction.
    humanize_cents:   optional small smoothed random pitch jitter
    max_correction_semitones: safety clamp - a legitimate scale correction is
                      never more than ~1 semitone, so anything bigger is a
                      detection error and must not be applied.

    Returns: shape (num_frames,) float32 ratios. Unvoiced frames get 1.0 when
    retune_ms == 0; with retune_ms > 0 the correction decays smoothly toward
    1.0 across unvoiced gaps instead of jumping.
    """
    detected = np.asarray(detected_pitches, dtype=np.float64)
    target = np.asarray(target_pitches, dtype=np.float64)
    num_frames = len(detected)
    voiced = (detected > 0) & (target > 0)

    error = np.zeros(num_frames)
    error[voiced] = 12.0 * np.log2(target[voiced] / detected[voiced])
    error = np.clip(error, -max_correction_semitones, max_correction_semitones)

    if confidence is None:
        weight = voiced.astype(np.float64)
    else:
        weight = np.where(voiced, np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0), 0.0)
    u = weight * error

    if retune_ms <= 0:
        correction = u
    else:
        hop_ms = 1000.0 * hop_size / sample_rate
        alpha = 1.0 - np.exp(-hop_ms / retune_ms)
        correction = np.zeros(num_frames)
        state = 0.0  # starts at "no correction": the natural onset
        for i in range(num_frames):
            state += alpha * (u[i] - state)
            correction[i] = state

    semitones = strength * correction

    if humanize_cents > 0.0:
        rng = np.random.default_rng(42)
        cents_noise = rng.uniform(-humanize_cents, humanize_cents, size=num_frames)
        smooth_jitter = np.convolve(cents_noise, np.ones(3) / 3.0, mode='same')
        semitones = semitones + np.where(voiced, smooth_jitter / 100.0, 0.0)

    return (2.0 ** (semitones / 12.0)).astype(np.float32)
