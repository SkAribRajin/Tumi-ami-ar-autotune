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


def denoise_audio(audio: np.ndarray, sample_rate: int, frame_size: int = 1024, hop_size: int = 256, over_subtraction: float = 1.8, noise_floor: float = 0.05) -> np.ndarray:
    """
    Spectral Subtraction Noise Reduction + Adaptive Noise Gate.
    Estimates stationary background noise floor (hiss, fan hum, AC noise, ambient room noise)
    from low-energy frames and subtracts it from the STFT magnitude spectrum.
    """
    if len(audio) < frame_size:
        return audio

    window = np.hanning(frame_size)
    num_frames = (len(audio) - frame_size) // hop_size + 1
    num_bins = frame_size // 2 + 1

    stft_mag = np.zeros((num_frames, num_bins), dtype=np.float32)
    stft_phase = np.zeros((num_frames, num_bins), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start : start + frame_size] * window
        spec = np.fft.rfft(frame)
        stft_mag[i] = np.abs(spec)
        stft_phase[i] = np.angle(spec)

    # Estimate noise spectrum from the quietest 15% energy frames
    frame_energies = np.mean(stft_mag ** 2, axis=1)
    lowest_k = max(1, int(num_frames * 0.15))
    quietest_indices = np.argsort(frame_energies)[:lowest_k]
    noise_spectrum = np.mean(stft_mag[quietest_indices], axis=0)

    # Spectral Subtraction
    cleaned_mag = np.zeros_like(stft_mag)
    for i in range(num_frames):
        subtracted = stft_mag[i] - over_subtraction * noise_spectrum
        floor = noise_floor * stft_mag[i]
        cleaned_mag[i] = np.maximum(subtracted, floor)

    # Adaptive Noise Gate for silence gaps
    vocal_envelope = np.mean(cleaned_mag, axis=1)
    gate_threshold = np.percentile(vocal_envelope, 20) * 1.5
    gate_mask = np.clip((vocal_envelope - gate_threshold) / (gate_threshold + 1e-6), 0.0, 1.0)
    smooth_mask = np.convolve(gate_mask, np.ones(5) / 5.0, mode='same')

    for i in range(num_frames):
        cleaned_mag[i] *= (0.15 + 0.85 * smooth_mask[i])

    # Inverse STFT & Overlap-Add
    output_len = (num_frames - 1) * hop_size + frame_size
    clean_audio = np.zeros(output_len, dtype=np.float32)
    win_sum = np.zeros(output_len, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        clean_spec = cleaned_mag[i] * np.exp(1j * stft_phase[i])
        frame_rec = np.fft.irfft(clean_spec, n=frame_size) * window
        clean_audio[start : start + frame_size] += frame_rec
        win_sum[start : start + frame_size] += window ** 2

    nonzero = win_sum > 1e-8
    clean_audio[nonzero] /= win_sum[nonzero]

    # Pad or trim to match exact input length
    if len(clean_audio) < len(audio):
        clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))
    else:
        clean_audio = clean_audio[:len(audio)]

    return clean_audio.astype(np.float32)

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



        



