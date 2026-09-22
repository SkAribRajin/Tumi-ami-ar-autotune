import numpy as np
from scipy.signal import butter, lfilter

"""
src/autotune/underwater.py
==========================
Enhanced "Underwater" voice effect module.
Implements:
1. Deep Muffling: 4th-order Butterworth Low-Pass Filter @ 480 Hz.
2. Sub-Aquatic Resonance: 260 Hz peaked resonant filter (Q=2.5).
3. Liquid Vocal Gurgle: Dual-rate amplitude tremolo (8.5 Hz & 14.2 Hz) while speaking.
4. Speech-Gated Bubbles: Vocal envelope follower triggers a dense stream of rising bubble chirps
   specifically while the user is speaking.
5. Perceived RMS Loudness Normalization: Preserves full volume matching the original input audio.
"""

def apply_deep_underwater_filter(audio: np.ndarray, sample_rate: int, cutoff: float = 480.0) -> np.ndarray:
    """
    4th-order steep Butterworth low-pass filter @ 480 Hz to heavily muffle the voice,
    removing all high-frequency crispness and mid-range brightness.
    """
    b, a = butter(4, cutoff, btype='lowpass', fs=sample_rate)
    return lfilter(b, a, audio).astype(np.float32)


def apply_underwater_resonance(audio: np.ndarray, sample_rate: int, center_freq: float = 260.0, Q: float = 2.5, gain_db: float = 6.0) -> np.ndarray:
    """
    Applies a resonant low-mid peak filter around 260 Hz to simulate the dense,
    enclosed acoustic resonance of sound traveling through water.
    """
    w0 = 2 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2 * Q)
    A = 10 ** (gain_db / 40.0)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])

    return lfilter(b, a, audio).astype(np.float32)


def apply_liquid_gurgle_tremolo(audio: np.ndarray, sample_rate: int, depth: float = 0.40) -> np.ndarray:
    """
    Applies a dual-rate liquid gurgling amplitude modulation (8.5 Hz & 14.2 Hz)
    that causes the voice to waver and gurgle as if speaking through water.
    """
    t = np.arange(len(audio)) / float(sample_rate)
    # Dual modulating frequencies create an irregular "bubbling voice" gurgle
    mod = 1.0 - depth * (0.6 * np.sin(2 * np.pi * 8.5 * t) + 0.4 * np.sin(2 * np.pi * 14.2 * t) + 1.0) / 2.0
    return (audio * mod).astype(np.float32)


def compute_vocal_envelope(audio: np.ndarray, sample_rate: int, frame_size: int = 512) -> np.ndarray:
    """
    Computes a smooth amplitude envelope E[t] of the vocal signal.
    Used to trigger bubbles specifically WHILE the user is speaking.
    """
    num_samples = len(audio)
    square = audio ** 2
    window = np.ones(frame_size) / frame_size
    rms = np.sqrt(np.maximum(0, np.convolve(square, window, mode='same')))
    max_val = np.max(rms)
    if max_val > 1e-6:
        return (rms / max_val).astype(np.float32)
    return rms.astype(np.float32)


def generate_speech_gated_bubbles(audio: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a dense stream of bubble chirps synchronized directly with the user's speech envelope.
    Bubbles intensify while speaking and quiet down during silence.
    """
    total_samples = len(audio)
    bubble_track = np.zeros(total_samples, dtype=np.float32)

    envelope = compute_vocal_envelope(audio, sample_rate)

    # Check speech energy in ~45ms hops
    hop = int(sample_rate * 0.045)
    
    for start_idx in range(0, total_samples - hop, hop):
        speech_energy = np.mean(envelope[start_idx : start_idx + hop])
        
        # Trigger bubbles proportional to speech energy
        if speech_energy > 0.06:
            prob = min(0.85, speech_energy * 1.4)
            if rng.random() < prob:
                b_dur = rng.uniform(0.035, 0.075)  # 35ms - 75ms rapid bubble chirps
                f_start = rng.uniform(350.0, 750.0)
                f_end = rng.uniform(800.0, 1650.0)
                amp = rng.uniform(0.20, 0.40) * speech_energy

                num_b_samples = int(b_dur * sample_rate)
                t_b = np.linspace(0, b_dur, num_b_samples, endpoint=False)
                # Upward pitch sweep (bubble rising)
                freq_curve = f_start + (f_end - f_start) * (t_b / b_dur) ** 2
                phase = 2 * np.pi * np.cumsum(freq_curve) / sample_rate
                env_b = np.exp(-7.0 * t_b / b_dur)
                bubble_wave = np.sin(phase) * env_b * amp

                end_idx = min(total_samples, start_idx + len(bubble_wave))
                insert_len = end_idx - start_idx
                bubble_track[start_idx:end_idx] += bubble_wave[:insert_len]

    return bubble_track


from .filters import denoise_audio

def apply_underwater_effect(audio: np.ndarray, sample_rate: int, seed: int = 42) -> np.ndarray:
    """
    One-click complete underwater effect.

    Parameters
    ----------
    audio : np.ndarray
        Input audio array (float32).
    sample_rate : int
        Audio sample rate.
    seed : int
        Random seed for deterministic bubble synthesis.

    Returns
    -------
    np.ndarray
        Processed underwater audio signal matching exact input length.
    """
    if len(audio) == 0:
        return audio

    # Denoise background noise first
    audio = denoise_audio(audio, sample_rate)

    rng = np.random.default_rng(seed)

    # Step 1: Deep Low-pass filter @ 480 Hz (heavily muffled voice)
    muffled_voice = apply_deep_underwater_filter(audio, sample_rate, cutoff=480.0)

    # Step 2: Sub-aquatic acoustic peak resonance @ 260 Hz
    resonant_voice = apply_underwater_resonance(muffled_voice, sample_rate, center_freq=260.0, Q=2.5, gain_db=6.0)

    # Step 3: Liquid vocal gurgle tremolo (wavering voice while speaking)
    gurgling_voice = apply_liquid_gurgle_tremolo(resonant_voice, sample_rate, depth=0.40)

    # Step 4: Speech-gated bubble generation (bubbles erupt WHILE speaking)
    speech_bubbles = generate_speech_gated_bubbles(audio, sample_rate, rng)

    # Step 5: Combine gurgling voice + speech-synchronous bubbles
    mixed = gurgling_voice * 0.90 + speech_bubbles * 0.40

    # Step 6: RMS perceived energy gain matching to raw input audio (full volume)
    rms_orig = np.sqrt(np.mean(audio ** 2))
    rms_mixed = np.sqrt(np.mean(mixed ** 2))
    if rms_mixed > 1e-6 and rms_orig > 1e-6:
        gain = rms_orig / rms_mixed
        mixed = mixed * gain

    # Step 7: Headroom safety clamp to prevent digital clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    return mixed.astype(np.float32)
