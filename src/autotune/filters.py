import numpy as np
from scipy.signal import lfilter
from scipy.signal import freqz

import matplotlib.pyplot as plt
from scipy.signal import tf2zpk

"""
Why do we need pre-emphasis?
A speech signal generally has more energy at lower 
frequencies than at higher frequencies.So high-frequency components of speech can be relatively weak.
For pitch detection, it can be useful to make the higher frequencies stronger.

pre emphasis equation: y[n] = x[n] - ax[n-1]
x[n] = current input sample
x[n-1] = previous input sample
y[n] = output sample
a = pre-emphasis coefficient

"""
def design_preemphasis_filter(coeff = 0.95):
    """
        A(z)=1
        H(z) = (1-0.95z^-1)/1
        H(z) = 1-0,95z^-1

        this is an FIR: FIR stands for Finite Impulse Response.
        It is a type of digital filter where the output depends on the current 
        and previous input samples, but not on previous output samples.

        FIR:

        y[n]=x[n]-0.95x[n-1] 

        Only previous inputs are used.

        IIR:

        y[n]=x[n]-0.95x[n-1]+0.5y[n-1] 
    """
    b = np.array([1.0, -coeff]) #b=[1, -0.95] coeffs of x[n-1] and x[n]
    a = np.array([1.0]) #denominator coefficient later needed for filtration and Z transform
    return b,a

from scipy.signal import butter

def design_underwater_filter(cutoff=800, sample_rate=44100):
    """
    Designs a 2nd-order Butterworth low-pass filter to simulate an
    'underwater' / muffled sub-aquatic acoustic effect by removing
    high frequencies above cutoff (default 800 Hz).
    """
    b, a = butter(2, cutoff, btype='lowpass', fs=sample_rate)
    return b, a

def apply_filter(audio, b, a):
    filtered = lfilter(b,a,audio)
    return filtered.astype(audio.dtype)


def denoise_audio(audio: np.ndarray, sample_rate: int, frame_size: int = 1024, hop_size: int = 256,
                  over_subtraction: float = 1.5, noise_floor: float = 0.1,
                  noise_percentile: float = 5.0, min_gap_contrast_db: float = 10.0,
                  clean_snr_db: float = 60.0) -> np.ndarray:
    """
    Stationary noise reduction (hiss, fan hum, room tone) with a Wiener-style
    spectral gain.

    What changed and why (see docs/diagnosis.md):
      * Noise estimate: the old code averaged the quietest 15% of frames
        whether or not they were quiet. A dense vocal (rap, belted chorus)
        has no silent frames, so that "noise" was actually voice; subtracting
        1.8x of it hollowed the voice out. The noise is now estimated only
        from real gaps, and if the track has none it is left alone.
      * The old gate (threshold = 1.5 x the 20th-percentile frame level)
        attenuated every frame near the typical level down to 0.15x on
        consistent-level material. It is removed; the Wiener gain below
        already suppresses gap noise.
      * Gain: G = sqrt(max(1 - a*N/|X|^2, floor^2)). Bins well above the
        noise get G ~ 1, so the voice is untouched. The floor limits damage
        if the estimate is wrong. The gain is smoothed over 3 frames to avoid
        "musical noise" (the watery/cloudy chirps of raw spectral subtraction).
      * If the quiet frames are more than clean_snr_db below the loud ones
        it's a clean studio stem and the input is returned untouched.
      * Edges are padded so the first/last samples aren't lost or distorted.
    """
    from scipy.ndimage import uniform_filter1d

    x = np.asarray(audio, dtype=np.float64)
    if len(x) < frame_size:
        return np.asarray(audio, dtype=np.float32)

    window = np.hanning(frame_size)
    padded = np.pad(x, (frame_size, frame_size + hop_size))
    num_frames = (len(padded) - frame_size) // hop_size + 1
    idx = np.arange(frame_size)[None, :] + hop_size * np.arange(num_frames)[:, None]
    spec = np.fft.rfft(padded[idx] * window, axis=1)
    power = np.abs(spec) ** 2

    # Noise PSD from genuine GAPS only: frames within 3 dB of the quietest 5%
    # of frames, and only if those are at least `min_gap_contrast_db` below the
    # loud frames. A dense rap/compressed vocal has no such gaps (its quiet
    # frames are still voice), and in that case stationary noise is masked by
    # the voice anyway, so we leave the audio untouched instead of subtracting
    # voice from itself (which measured -9 dB on a dense synthetic rap line).
    frame_db = 10 * np.log10(power.sum(axis=1) + 1e-20)
    quiet_db = np.percentile(frame_db, noise_percentile)
    loud_db = np.percentile(frame_db, 95)
    if loud_db - quiet_db < min_gap_contrast_db or loud_db - quiet_db > clean_snr_db:
        return np.asarray(audio, dtype=np.float32)
    gaps = frame_db <= quiet_db + 3.0
    noise_psd = power[gaps].mean(axis=0)

    gain_sq = 1.0 - over_subtraction * noise_psd[None, :] / np.maximum(power, 1e-20)
    gain = np.sqrt(np.maximum(gain_sq, noise_floor ** 2))
    gain = uniform_filter1d(gain, size=3, axis=0)

    frames_out = np.fft.irfft(spec * gain, n=frame_size, axis=1) * window
    out = np.zeros(len(padded))
    win_sum = np.zeros(len(padded))
    for i in range(num_frames):
        s = i * hop_size
        out[s:s + frame_size] += frames_out[i]
        win_sum[s:s + frame_size] += window ** 2
    nonzero = win_sum > 1e-8
    out[nonzero] /= win_sum[nonzero]

    return out[frame_size:frame_size + len(x)].astype(np.float32)


def soft_limit(audio: np.ndarray, sample_rate: int, ceiling: float = 0.98,
               lookahead_ms: float = 2.5) -> np.ndarray:
    """
    Transparent peak limiter that replaces the old "scale the WHOLE file so
    its single highest peak is 0.95" step, which let one transient set the
    loudness of the entire song.

    g[n] = min(1, ceiling/|x[n]|) is the gain each sample needs. A minimum
    filter over +/-L samples followed by a smoothing kernel no wider than
    +/-L keeps the gain curve smooth (no distortion) and guarantees
    g_smooth[n] <= g[n], so |output| <= ceiling. Only the few milliseconds
    around an over are turned down; the rest of the file is untouched.
    """
    from scipy.ndimage import minimum_filter1d

    x = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak <= ceiling:
        return np.asarray(audio, dtype=np.float32)
    L = max(1, int(sample_rate * lookahead_ms / 1000.0))
    need = np.minimum(1.0, ceiling / np.maximum(np.abs(x), 1e-12))
    held = minimum_filter1d(need, size=2 * L + 1, mode='nearest')
    kernel = np.hanning(2 * L + 3)[1:-1]
    kernel /= kernel.sum()
    smooth = np.convolve(np.pad(held, L, mode='edge'), kernel, mode='valid')
    return (x * smooth).astype(np.float32)

def plot_pole_zero(b, a, save_path):
    zeros, poles, _ = tf2zpk(b,a)

    fig, ax = plt.subplots(figsize=(5,5))

    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linestyle = "--", color="gray")

    ax.scatter(zeros.real, zeros.imag, marker="o", facecolors="none", edgecolors="blue", s=80, label="Zeros")
    ax.scatter(poles.real, poles.imag, marker="x", color="red", s=80, label="Poles")

    ax.axhline(0, color="black", linewidth=.5)
    ax.axvline(0, color="black", linewidth=.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Pole-Zero Plot")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_frequency_response(b, a, sample_rate, save_path):
    w, h = freqz(b,a, worN=2048, fs=sample_rate)

    magnitude_db = 20*np.log10(np.abs(h)+1e-12)
    phase_deg = np.unwrap(np.angle(h))*(180/np.pi)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6), sharex=True)

    ax1.plot(w, magnitude_db, color="#2980b9")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Frequency Response")
    ax1.grid(True, alpha=0.3)

    ax2.plot(w, phase_deg, color="#e67e22")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)



        



