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

def compute_shift_ratios(detected_pitches, target_pitches, strength=1.0,
                         preserve_vibrato=True, onset_protection_ms=30.0,
                         humanize_cents=0.0, hop_size=512, sample_rate=44100):
    """
    detected_pitches: shape (num_frames,) - detected pitch in Hz per frame
    target_pitches:   shape (num_frames,) - target scale note in Hz per frame
    strength:         float 0.0 to 1.0 - correction strength
    preserve_vibrato: bool - if True, preserves natural 5-7 Hz vibrato oscillations around target note
    onset_protection_ms: float - duration in ms to protect note attack pitch-bends
    humanize_cents:   float - small cents jitter (e.g. 3.0) for natural organic variation
    """
    num_frames = len(detected_pitches)
    ratios = np.ones(num_frames, dtype=np.float32)
    voiced = detected_pitches > 0

    if not np.any(voiced):
        return ratios

    raw_ratio = np.ones(num_frames, dtype=np.float32)
    raw_ratio[voiced] = target_pitches[voiced] / detected_pitches[voiced]

    effective_target = np.copy(target_pitches)

    # 1. Vibrato Handling: Preserve 5-7 Hz pitch oscillations around rolling center
    if preserve_vibrato and num_frames > 5:
        # 200ms rolling window for vibrato center estimation
        window_len = max(3, int(0.200 * sample_rate / hop_size))
        rolling_mean = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            if not voiced[i]:
                continue
            start_i = max(0, i - window_len // 2)
            end_i = min(num_frames, i + window_len // 2 + 1)
            v_in_win = voiced[start_i:end_i]
            if np.any(v_in_win):
                rolling_mean[i] = np.mean(detected_pitches[start_i:end_i][v_in_win])
            else:
                rolling_mean[i] = detected_pitches[i]

        for i in range(num_frames):
            if voiced[i] and rolling_mean[i] > 0:
                vibrato_ratio = detected_pitches[i] / rolling_mean[i]
                # Preserve vibrato fluctuation around target note
                effective_target[i] = target_pitches[i] * vibrato_ratio

    # 2. Note-Onset Protection: Protect initial milliseconds of a new note attack
    hop_time_ms = (hop_size / sample_rate) * 1000.0
    onset_frames = int(round(onset_protection_ms / hop_time_ms)) if hop_time_ms > 0 else 0

    effective_strength = np.full(num_frames, strength, dtype=np.float32)
    if onset_frames > 0:
        voiced_run_len = 0
        for i in range(num_frames):
            if voiced[i]:
                voiced_run_len += 1
                if voiced_run_len <= onset_frames:
                    # Gradually ramp strength from 0 to full strength during note onset
                    ramp = voiced_run_len / float(onset_frames)
                    effective_strength[i] = strength * ramp
            else:
                voiced_run_len = 0

    # Calculate ratios with effective strength and target
    for i in range(num_frames):
        if voiced[i] and detected_pitches[i] > 0:
            target_f = effective_target[i]
            base_r = target_f / detected_pitches[i]
            ratios[i] = base_r ** effective_strength[i]

    # 3. Micro-pitch Humanization (cents jitter)
    if humanize_cents > 0.0:
        rng = np.random.default_rng(42)
        # 1 semitone = 100 cents -> cent factor = 2^(cents / 1200)
        cents_noise = rng.uniform(-humanize_cents, humanize_cents, size=num_frames)
        # Low-pass filter noise for smooth organic micro-fluctuation
        smooth_jitter = np.convolve(cents_noise, np.ones(3)/3.0, mode='same')
        jitter_ratios = 2.0 ** (smooth_jitter / 1200.0)
        ratios[voiced] *= jitter_ratios[voiced]

    return ratios.astype(np.float32)



def smooth_shift_ratios(shift_ratios, detected_pitches, hop_size, sample_rate, retune_ms=40.0):
    """
    Smooths the per-frame correction ratio over time so pitch glides toward
    the target note instead of snapping fully in a single ~11ms frame. This
    is what separates a natural-sounding correction from the hard, robotic
    "T-Pain" snap - compute_shift_ratios() alone recomputes a fresh target
    every frame with no memory of the previous frame, so any jitter in the
    detected pitch (very normal with autocorrelation + natural vibrato)
    shows up directly as flutter in the corrected audio.

    Uses an exponential moving average in the LOG of the ratio, not the raw
    ratio - shift ratios are multiplicative (a ratio of 2.0 up and 0.5 down
    are equally "one octave"), so averaging in log space is what keeps the
    glide symmetric between upward and downward corrections.

    The average resets at the start of every voiced run (i.e. after a
    silence/unvoiced gap) so a new note starts clean instead of gliding in
    from whatever the previous note's ratio happened to be.

    retune_ms: how many milliseconds it takes to glide most of the way to
    the target pitch.
      0        -> no smoothing at all (identical to the old instant-snap behavior)
      ~30-80   -> natural-sounding correction
      100+     -> audible pitch bends/glides rather than a "correction"
    """
    if retune_ms <= 0:
        return shift_ratios

    log_ratios = np.log(shift_ratios.astype(np.float64))
    smoothed_log = np.zeros_like(log_ratios)

    hop_time_ms = (hop_size / sample_rate) * 1000.0
    alpha = 1.0 - np.exp(-hop_time_ms / retune_ms)

    prev = None
    for i in range(len(log_ratios)):
        if detected_pitches[i] <= 0:
            # unvoiced: nothing to glide, and reset so the next note doesn't
            # inherit a stale glide-in-progress from before the gap
            smoothed_log[i] = log_ratios[i]
            prev = None
            continue
        if prev is None:
            smoothed_log[i] = log_ratios[i]  # first voiced frame of a run: start clean, no glide-in
        else:
            smoothed_log[i] = prev + alpha * (log_ratios[i] - prev)
        prev = smoothed_log[i]

    return np.exp(smoothed_log).astype(np.float32)