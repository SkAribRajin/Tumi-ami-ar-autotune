import sys
sys.path.insert(0, '.')
import numpy as np

"""
Phase Vocoder Pitch Shifting
==============================
See project chat history / report for full worked-example explanations of
each piece. Summary:
  1. zero-phase STFT of each frame (magnitude + phase at the frame centre)
  2. find_peaks / assign_regions - locate spectral peaks (harmonics) and figure
     out which bins "belong" to which peak
  3. per peak: true frequency from the phase advance between frames, bin
     offset for the new frequency, and one phase rotation for the whole region
  4. phase_vocoder_shift - ties it all together (the CONTRACTS.md function),
     with optional ratio-aware formant preservation
  (The history below explains why energy has to be MOVED between bins; the
  "Rewrite notes" further down explain what was still wrong with the first
  fix and how the current version handles it.)

--------------------------------------------------------------------------
Why this version differs from a naive "keep magnitude in the same bin, just
push the phase" approach (the bug we found and fixed):

Each FFT bin behaves like a narrow bandpass filter centered on that bin's own
frequency (bandwidth roughly +/- sample_rate/frame_size). If you leave a
harmonic's magnitude sitting in its original bin but tell its phase to evolve
as if it were now a much higher/lower frequency, you're asking that narrow
filter to output energy far outside the range it's physically capable of
producing - the phase trick can't overcome that. That's why big shift ratios
(1.5x, 2x) didn't just sound bad, they collapsed toward the wrong pitch
entirely.

The fix has two parts:
  (a) Actually move each harmonic's energy to a bin near its NEW (shifted)
      frequency, instead of leaving it in place. This is the frequency-domain
      equivalent of "resample the spectrum" - same physical effect as
      resampling the waveform, but done per-bin so it still fits the
      existing fixed-hop frame-in/frame-out contract instead of needing a
      separate whole-signal time-stretch+resample pass.
  (b) Phase-lock the bins around each harmonic peak together (rather than
      letting every bin accumulate its own independent phase trajectory).
      Small per-bin frequency-estimate errors used to accumulate over many
      frames and pull neighboring bins out of sync with each other - that's
      what caused the smearing/interference even at moderate ratios like
      1.05. Locking them to their peak means only one phase accumulator per
      harmonic can drift, not one per bin.
--------------------------------------------------------------------------
"""

"""
Rewrite notes (see docs/diagnosis.md for the measurements):

The previous remap_and_lock_spectrum kept ONE running phase per OUTPUT bin
and only advanced it on frames where a peak happened to land in that bin.
When a peak moved by one bin between frames (which happens with every bit
of vibrato and every ratio change) it picked up a phase that was stale by
several hops. Result: at ratio 1.0 the output had an SNR of -2 to -3 dB
against the input, i.e. the phase was effectively scrambled. That is the
"cloudy" sound. It also merged colliding bins with max() instead of adding
them (loses energy), and ignored harmonics quieter than -26 dB, so their
bins were shifted with a lower harmonic's offset (inharmonic smear).

This version is Laroche & Dolson's (1999) phase-locked pitch shifter:
  * zero-phase analysis (frame rotated by N/2 before the FFT) so the phase
    of a peak bin is the sinusoid's phase at the FRAME CENTRE; all main-lobe
    bins of a peak then share that phase
  * every local maximum is a peak; each bin belongs to its nearest peak
  * each peak region is moved by an integer number of bins
        dk = round((ratio - 1) * true_frequency_in_bins)
    and all its bins are multiplied by ONE complex rotation exp(j*theta)
    (identity phase locking)
  * the rotation is tracked per partial, not per output bin:
        theta_i = theta_{i-1} + (ratio - 1) * Omega
    where Omega is the partial's measured phase advance per hop. Ratio 1
    therefore gives theta = 0 and the output is bit-identical to the input
  * colliding bins are summed (complex), so energy is preserved
"""

def wrap_phase(phase_diff):
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


def find_peaks(magnitude, min_rel_height=1e-5):
    """
    Local maxima of the magnitude spectrum. Every local maximum counts (down
    to -100 dB relative to the frame's loudest bin), including weak upper
    harmonics and noise. Each is shifted by its own frequency-proportional
    amount, which is what keeps the upper harmonics harmonic.
    """
    num_bins = len(magnitude)
    if num_bins < 3:
        return np.array([0], dtype=int)
    peak_max = magnitude.max()
    if peak_max <= 1e-12:
        return np.array([int(np.argmax(magnitude))])
    mask = np.zeros(num_bins, dtype=bool)
    mask[1:-1] = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:])
    mask &= magnitude > (min_rel_height * peak_max)
    peaks = np.flatnonzero(mask)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(magnitude))])
    return peaks


def assign_regions(peaks, num_bins):
    """
    For every bin, which peak "owns" it (region of influence = nearer peak,
    split at the midpoint between adjacent peaks).
    """
    if len(peaks) == 0:
        return np.zeros(num_bins, dtype=int)
    boundaries = (peaks[:-1] + peaks[1:]) / 2.0
    return peaks[np.searchsorted(boundaries, np.arange(num_bins))]


def spectral_envelope_log(magnitude, frame_size, num_coeffs=30):
    """
    Log spectral envelope (formant shape) of one magnitude spectrum via
    cepstral liftering: keep the first num_coeffs quefrency samples. 30
    samples at 44.1 kHz is 0.68 ms, well below the shortest pitch period
    we track (1100 Hz -> 0.9 ms), so harmonics don't leak into the envelope.
    """
    cepstrum = np.fft.irfft(np.log(magnitude + 1e-9), n=frame_size)
    cepstrum[num_coeffs:frame_size - num_coeffs + 1] = 0.0
    return np.fft.rfft(cepstrum, n=frame_size).real


def reconstruct_frame(spectrum, config):
    """Complex (zero-phase) spectrum -> time-domain frame in normal sample order."""
    frame = np.fft.irfft(spectrum, n=config.frame_size)
    return np.roll(frame, config.frame_size // 2).astype(np.float32)


def phase_vocoder_shift(frames, shift_ratios, config, preserve_formants=False,
                        formant_coeffs=30, max_formant_gain_db=12.0):
    """
    frames: (num_frames, frame_size) Hann-windowed frames from frame_signal()
    shift_ratios: (num_frames,) one pitch ratio per frame
    preserve_formants: if True, re-imposes the ORIGINAL frame's spectral
        envelope on the shifted spectrum. Because we know the ratio, the
        envelope the shift produced is just env(k / ratio), so the gain is
        env(k) / env(k / ratio). No need to estimate an envelope from the gappy
        shifted spectrum, which is what made the old formant_preserve apply
        +4..+22 dB gain swings per frame.

    Returns (num_frames, frame_size) float32 frames for overlap_add().
    """
    N = config.frame_size
    H = config.hop_size
    num_frames = frames.shape[0]
    num_bins = N // 2 + 1
    bins = np.arange(num_bins)
    expected_advance = 2 * np.pi * bins * H / N
    max_gain = np.log(10 ** (max_formant_gain_db / 20.0))

    out = np.zeros((num_frames, N), dtype=np.float32)
    prev_phase = None
    theta_by_bin = np.zeros(num_bins)  # rotation applied last frame, per input bin

    for i in range(num_frames):
        X = np.fft.rfft(np.roll(frames[i].astype(np.float64), -(N // 2)))
        mag = np.abs(X)
        phase = np.angle(X)
        ratio = float(shift_ratios[i])

        peaks = find_peaks(mag)
        owner = assign_regions(peaks, num_bins)

        # measured phase advance per hop of each peak (radians) = its true frequency
        if prev_phase is None:
            omega = expected_advance[peaks]
        else:
            omega = expected_advance[peaks] + wrap_phase(
                phase[peaks] - prev_phase[peaks] - expected_advance[peaks])

        theta_peaks = wrap_phase(theta_by_bin[peaks] + (ratio - 1.0) * omega)
        freq_in_bins = omega * N / (2 * np.pi * H)
        dk_peaks = np.round((ratio - 1.0) * freq_in_bins).astype(int)

        theta_full = np.zeros(num_bins)
        dk_full = np.zeros(num_bins, dtype=int)
        theta_full[peaks] = theta_peaks
        dk_full[peaks] = dk_peaks
        theta = theta_full[owner]
        dest = bins + dk_full[owner]

        if not np.any(dk_peaks) and not np.any(theta_peaks):
            Y = X  # nothing to do: exact identity
        else:
            rotated = X * np.exp(1j * theta)
            ok = (dest >= 0) & (dest < num_bins)
            Y = (np.bincount(dest[ok], weights=rotated.real[ok], minlength=num_bins)
                 + 1j * np.bincount(dest[ok], weights=rotated.imag[ok], minlength=num_bins))

        if preserve_formants and ratio != 1.0:
            log_env = spectral_envelope_log(mag, N, formant_coeffs)
            warped = np.interp(bins / ratio, bins, log_env)
            Y = Y * np.exp(np.clip(log_env - warped, -max_gain, max_gain))

        out[i] = reconstruct_frame(Y, config)
        prev_phase = phase
        theta_by_bin = theta

    return out


"""
Formant Preservation
=======================
Problem: when the phase vocoder shifts pitch, it also drags your voice's
natural resonances (formants) along with it - a large shift makes you sound
like you inhaled helium ("chipmunk effect"), because the ENTIRE spectral
shape moved, not just the pitch.

The fix separates two things that are tangled together in a voice signal:
  - PITCH: which note is being sung (fast-changing, the harmonic "comb")
  - FORMANTS: what your voice naturally sounds like (slow-changing, the
    overall spectral SHAPE/envelope that harmonics sit inside)

We want to shift pitch but keep formants fixed. The technique below
("cepstral smoothing") separates them using the same FFT tools you already
trust:

    take the log of the magnitude spectrum
    -> inverse FFT it (this gives something called a "cepstrum")
    -> keep only the first ~30 values, zero out the rest
    -> FFT back

This works because in this transformed domain, the SLOW-changing formant
shape lives in the first few coefficients, and the FAST-changing pitch
harmonic detail lives in the later ones - so keeping only the first ~30
acts like a low-pass filter that isolates just the formant shape.
"""

def compute_spectral_envelope(frame, config, num_coeffs=20):
    """
    Extracts the smooth "formant shape" of a frame, discarding the sharp
    pitch-harmonic detail.

    frame: np.ndarray, shape (frame_size,) - a time-domain audio frame
    num_coeffs: how many low-quefrency cepstral coefficients to keep.
                Smaller = smoother envelope (warm, sweet, no metallic artifacts).
                ~18-24 gives ultra-smooth human vocal warmth.
    """
    spectrum = np.fft.rfft(frame)
    log_magnitude = np.log(np.abs(spectrum) + 1e-8)  # +1e-8 avoids log(0)

    # Inverse FFT of the LOG magnitude gives the "cepstrum"
    cepstrum = np.fft.irfft(log_magnitude, n=config.frame_size)

    # Liftering: zero out everything except the first and last few
    liftered = np.zeros_like(cepstrum)
    liftered[:num_coeffs] = cepstrum[:num_coeffs]
    liftered[-(num_coeffs - 1):] = cepstrum[-(num_coeffs - 1):]

    # Transform back: this is now a SMOOTHED version of the log spectrum
    smooth_log_magnitude = np.fft.rfft(liftered, n=config.frame_size).real
    envelope = np.exp(smooth_log_magnitude)
    return envelope


def formant_preserve(shifted_frame, original_frame, config, num_coeffs=20):
    """
    Corrects a pitch-shifted frame so it keeps the ORIGINAL frame's formant
    shape instead of the shifted frame's (accidentally moved) formant shape.

    LEGACY - kept for the CONTRACTS.md signature, no longer used by
    run_pipeline. It estimates the shifted envelope from the shifted frame
    itself; when the shift leaves gaps between harmonic regions, log(~0) in
    those gaps drags that envelope down and this function boosts the frame
    by up to +22 dB (measured). The pipeline uses
    phase_vocoder_shift(..., preserve_formants=True), which uses the known
    ratio instead.

    shifted_frame: np.ndarray, shape (frame_size,) - output of the phase
                   vocoder for this frame (pitch already shifted)
    original_frame: np.ndarray, shape (frame_size,) - the SAME frame before
                     shifting (the natural, correct-formant version)
    config: AutoTuneConfig

    Returns: np.ndarray, shape (frame_size,) - the shifted frame, with its
    formant shape corrected back to match the original voice's natural shape.
    """
    # Step 1: what formant shape did we START with? (the one we want to keep)
    orig_envelope = compute_spectral_envelope(original_frame, config, num_coeffs)

    # Step 2: break the SHIFTED frame into magnitude + phase (same as
    # compute_stft in the phase vocoder - we need both pieces separately)
    shifted_spectrum = np.fft.rfft(shifted_frame)
    shifted_magnitude = np.abs(shifted_spectrum)
    shifted_phase = np.angle(shifted_spectrum)

    # Step 3: what formant shape did the SHIFTED frame accidentally end up
    # with? (this is the "wrong" shape we want to remove)
    shifted_envelope = compute_spectral_envelope(shifted_frame, config, num_coeffs)

    # Step 4: remove the shifted frame's own (wrong) envelope by dividing
    # it out, then multiply in the original (correct) envelope instead.
    # This keeps the shifted frame's actual pitch/harmonics exactly as they
    # are - only the overall SHAPE draped over them changes.
    corrected_magnitude = shifted_magnitude / (shifted_envelope + 1e-8) * orig_envelope

    # Step 5: rebuild the frame using the corrected magnitude, but the
    # SHIFTED frame's phase (we want to keep the new pitch's timing, only
    # the tonal shape is being corrected)
    corrected_spectrum = corrected_magnitude * (np.cos(shifted_phase) + 1j * np.sin(shifted_phase))
    frame = np.fft.irfft(corrected_spectrum, n=config.frame_size)
    return frame.astype(np.float32)


'''
import sys
sys.path.insert(0, '.')
import numpy as np


"""
# Piece 1: Converting frames to frequency-domain (magnitude + phase)

Phase Vocoder Pitch Shifting — Overview
=========================================

Recall from pitch_detection.py: np.fft.rfft(frame) gives us a COMPLEX array,
one complex number per frequency bin. Each complex number encodes TWO things:

    magnitude = np.abs(X[k])        -> "how much" of this frequency is present
    phase     = np.angle(X[k])      -> "what point in its cycle" this frequency
                                        was at, at the start of this frame

For pitch detection, we only used magnitude (via the power spectrum). But to
properly SHIFT pitch without destroying the sound, we need both. Here's why:

If we simply move energy from bin k to bin k*ratio (to shift pitch), but
reconstruct each frame's phase independently (e.g., starting from 0 every
time), the frames won't line up smoothly when we overlap-add them back
together. Consecutive frames need their phases to evolve smoothly and
consistently, or you get a robotic/buzzy "phasiness" artifact.

So step 1 (this piece): extract magnitude AND phase from every frame,
so we have the full picture to work with before we touch anything.
"""

def compute_stft(frames, config):
    """
    frames: np.ndarray, shape (num_frames, frame_size) - already windowed,
            this is exactly the output of frame_signal() from framing.py

    Returns two arrays, both shape (num_frames, frame_size//2 + 1):
        magnitudes: how much energy is in each frequency bin, per frame
        phases:     the phase angle (in radians) of each frequency bin, per frame

    Note: frame_size//2 + 1 is the number of bins rfft gives us for a
    real-valued input of length frame_size (same reasoning you already
    saw in pitch_detection.py's use of np.fft.rfft).
    """
    num_frames = frames.shape[0]
    num_bins = config.frame_size // 2 + 1

    magnitudes = np.zeros((num_frames, num_bins))
    phases = np.zeros((num_frames, num_bins))

    for i in range(num_frames):
        spectrum = np.fft.rfft(frames[i])
        magnitudes[i] = np.abs(spectrum)
        phases[i] = np.angle(spectrum)

    return magnitudes, phases









"""
Piece 2: Instantaneous Frequency via Phase Difference
========================================================

Why we need this: bin frequencies are coarse (locked to frame_size/sample_rate
spacing, ~21.5 Hz apart at our settings). Real voice pitch can be anywhere,
not just at these exact bin centers. By comparing how much the phase actually
advanced between two consecutive frames vs. how much we'd EXPECT it to
advance if the signal were sitting exactly at the bin's center frequency,
we can figure out the TRUE frequency much more precisely.

Worked numeric example (frame_size=2048, hop_size=512, sample_rate=44100):
    bin k=20 center frequency = 20 * 44100/2048 = 430.7 Hz
    expected phase advance per hop for this bin:
        = 2*pi * k * hop_size / frame_size
        = 2*pi * 20 * 512/2048
        = 2*pi * 5.0
        = 31.416 radians

    Suppose the ACTUAL signal is exactly 440 Hz (slightly higher than the
    bin's 430.7 Hz center). Its true phase advance per hop would be:
        = 2*pi * 440 * (512/44100)
        = 2*pi * 5.1043
        = 32.076 radians

    deviation = actual - expected = 32.076 - 31.416 = 0.660 radians per hop
    extra frequency = deviation / (2*pi) * (sample_rate/hop_size)
                     = 0.660 / (2*pi) * (44100/512)
                     = 0.105 * 86.13
                     = 9.04 Hz

    true frequency estimate = bin center + extra = 430.7 + 9.04 = 439.7 Hz
    -> much closer to the real 440 Hz than the coarse 430.7 Hz bin estimate!

The phase-wrapping problem: np.angle() always returns values in [-pi, pi].
Our raw phase difference (actual - expected) might genuinely be, say, 7.5
radians, but np.angle()-based subtraction will show something like
7.5 - 2*pi = 1.2 instead. We MUST correct for this by wrapping the deviation
back into [-pi, pi] ourselves before converting it to a frequency - otherwise
our "extra frequency" calculation above would be wildly wrong.
"""

def wrap_phase(phase_diff):
    """
    Wraps a phase difference (in radians) into the range [-pi, pi].

    Example: wrap_phase(7.5) 
        7.5 is more than pi (3.14) so it's "too big" - it actually represents
        the same angle as 7.5 - 2*pi = 1.22 radians
    Example: wrap_phase(-4.0)
        -4.0 is less than -pi, so it wraps to -4.0 + 2*pi = 2.28 radians
    """
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


def compute_instantaneous_frequency(phases, config):
    """
    phases: np.ndarray, shape (num_frames, num_bins) - output of compute_stft()
    config: AutoTuneConfig - needs hop_size, frame_size, sample_rate

    Returns: np.ndarray, shape (num_frames, num_bins) - precise frequency (Hz)
    estimate for every bin, in every frame.

    Note: the very first frame has no "previous frame" to compare against,
    so we just use the bin's center frequency as a reasonable starting guess.
    """
    num_frames, num_bins = phases.shape
    freqs = np.zeros((num_frames, num_bins))

    # bin_center_freqs[k] = the "coarse" frequency that bin k represents
    bin_center_freqs = np.arange(num_bins) * config.sample_rate / config.frame_size

    # expected phase advance per hop, if a signal sat exactly at each bin's center
    expected_advance = 2 * np.pi * np.arange(num_bins) * config.hop_size / config.frame_size

    freqs[0] = bin_center_freqs  # first frame: no previous phase to compare, use coarse estimate

    for i in range(1, num_frames):
        actual_advance = phases[i] - phases[i - 1]
        deviation = wrap_phase(actual_advance - expected_advance)

        # convert the deviation (radians per hop) into an extra frequency (Hz)
        extra_freq = deviation / (2 * np.pi) * (config.sample_rate / config.hop_size)

        freqs[i] = bin_center_freqs + extra_freq

    return freqs





"""
Piece 3: Shifting Frequency and Re-Synthesizing Phase
=========================================================

Now that we know the TRUE frequency in each bin (from Piece 2), shifting
pitch is conceptually simple: multiply every bin's true frequency by our
shift_ratio. E.g., if shift_ratio=1.05, a bin whose true content was 440 Hz
now represents content at 462 Hz.

But we can't just "put" this shifted frequency into a fresh phase from
scratch each frame - remember, that causes robotic artifacts. Instead we
must ACCUMULATE phase smoothly across frames, using the shifted frequency.

Think of a phase accumulator as a runner going around a track:
    new_phase = previous_synthesized_phase + (shifted_frequency's expected
                phase advance for this hop)

This is exactly like: position = previous_position + speed * time
Except here "speed" is angular frequency and "time" is hop_size/sample_rate.

Worked example:
    true frequency = 440 Hz, shift_ratio = 1.05
    shifted frequency = 440 * 1.05 = 462 Hz

    phase advance per hop for 462 Hz:
        = 2*pi * 462 * (hop_size/sample_rate)
        = 2*pi * 462 * (512/44100)
        = 2*pi * 5.363
        = 33.70 radians

    if previous synthesized phase (for this bin) was, say, 1.0 radian:
        new synthesized phase = 1.0 + 33.70 = 34.70 radians
    (we do NOT wrap this - phase accumulates continuously across the
    whole signal, we only wrap when actually reading with np.sin/cos
    which handles huge angles fine automatically)
"""

def shift_and_resynthesize_phase(true_freqs, shift_ratios, config):
    """
    true_freqs: np.ndarray, shape (num_frames, num_bins) - output of
                compute_instantaneous_frequency()
    shift_ratios: np.ndarray, shape (num_frames,) - one ratio per frame,
                  output of compute_shift_ratios() from pitch_shift.py
    config: AutoTuneConfig - needs hop_size, sample_rate

    Returns: np.ndarray, shape (num_frames, num_bins) - new synthesized
    phase for every bin, in every frame, accumulated smoothly across time.
    """
    num_frames, num_bins = true_freqs.shape
    synthesized_phase = np.zeros((num_frames, num_bins))

    for i in range(num_frames):
        # shift this frame's true frequencies by this frame's shift ratio
        shifted_freqs = true_freqs[i] * shift_ratios[i]

        # how much phase this shifted frequency should advance in one hop
        phase_advance = 2 * np.pi * shifted_freqs * (config.hop_size / config.sample_rate)

        if i == 0:
            synthesized_phase[i] = phase_advance  # start accumulating from 0
        else:
            synthesized_phase[i] = synthesized_phase[i - 1] + phase_advance

    return synthesized_phase






"""
Piece 4: Reconstructing a Time-Domain Frame
==============================================

We now have, for each bin, in each frame:
    magnitude       - unchanged from the original (we keep the same "tone color")
    synthesized phase - our new smoothly-accumulated, shifted-pitch phase

A complex number can be built from magnitude and phase using Euler's formula:
    complex_value = magnitude * (cos(phase) + j*sin(phase))
    (equivalently: magnitude * exp(j*phase), same thing)

Once we rebuild the full complex spectrum this way, np.fft.irfft() converts
it back into a real time-domain frame - exactly the reverse of np.fft.rfft()
we used in Piece 1.
"""

def reconstruct_frame(magnitude, synthesized_phase, config):
    """
    magnitude: np.ndarray, shape (num_bins,) - one frame's magnitude spectrum
    synthesized_phase: np.ndarray, shape (num_bins,) - one frame's new phase
    config: AutoTuneConfig - needs frame_size

    Returns: np.ndarray, shape (frame_size,) - reconstructed time-domain frame
    """
    complex_spectrum = magnitude * (np.cos(synthesized_phase) + 1j * np.sin(synthesized_phase))
    frame = np.fft.irfft(complex_spectrum, n=config.frame_size)
    return frame.astype(np.float32)






"""
Piece 5: The Full Phase Vocoder Pipeline
===========================================

This ties Pieces 1-4 together into one function matching the signature
agreed in CONTRACTS.md:

    phase_vocoder_shift(frames, shift_ratios, config) -> shifted_frames

Workflow per frame:
    1. Convert all frames to magnitude + phase        (Piece 1)
    2. Get precise true frequency per bin, per frame   (Piece 2)
    3. Shift frequencies + accumulate new phase        (Piece 3)
    4. Rebuild each frame from magnitude + new phase   (Piece 4)
"""

def phase_vocoder_shift(frames, shift_ratios, config):
    """
    frames: np.ndarray, shape (num_frames, frame_size) - output of frame_signal()
    shift_ratios: np.ndarray, shape (num_frames,) - output of compute_shift_ratios()
    config: AutoTuneConfig

    Returns: np.ndarray, shape (num_frames, frame_size) - shifted frames,
    same shape as input, ready to pass straight into overlap_add()
    """
    magnitudes, phases = compute_stft(frames, config)
    true_freqs = compute_instantaneous_frequency(phases, config)
    new_phases = shift_and_resynthesize_phase(true_freqs, shift_ratios, config)

    num_frames = frames.shape[0]
    shifted_frames = np.zeros((num_frames, config.frame_size), dtype=np.float32)

    for i in range(num_frames):
        shifted_frames[i] = reconstruct_frame(magnitudes[i], new_phases[i], config)

    return shifted_frames




                                        #Template

if __name__ == "__main__":
    from src.autotune.config import AutoTuneConfig
    from src.autotune.io_utils import load_audio, save_audio
    from src.autotune.framing import frame_signal, overlap_add
    from src.autotune.pitch_detection import detect_pitch_for_all_frames
    from src.autotune.scales import build_scale_midi_set, nearest_scale_note
    from src.autotune.pitch_shift import compute_shift_ratios

    config = AutoTuneConfig()
    audio, sr = load_audio("data/raw/test_voice.wav", target_sr=config.sample_rate)
    frames, pad_len = frame_signal(audio, config)

    detected = detect_pitch_for_all_frames(frames, sr)
    scale = build_scale_midi_set("C", "major")
    target = np.array([nearest_scale_note(f, scale) if f > 0 else 0.0 for f in detected])
    ratios = compute_shift_ratios(detected, target, strength=1.0)

    shifted_frames = phase_vocoder_shift(frames, ratios, config)

    output = overlap_add(shifted_frames, config, pad_len)
    save_audio("data/processed/phase_vocoder_shifted.wav", output, sr)
    print("Saved data/processed/phase_vocoder_shifted.wav")
    print("Compare this against naive_shifted.wav - this should sound noticeably cleaner!")

    '''