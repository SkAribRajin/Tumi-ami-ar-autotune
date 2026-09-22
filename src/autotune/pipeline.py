# this was created  Week 5-6: full pipeline + customization features (scale selector, correction strength)


import numpy as np
from .io_utils import load_audio, save_audio
from .framing import frame_signal, overlap_add
from .pitch_detection import detect_pitch_for_all_frames
from .scales import build_scale_midi_set, nearest_scale_note
from .pitch_shift import compute_shift_ratios, smooth_shift_ratios, naive_pitch_shift
from .phase_vocoder import phase_vocoder_shift
from .filters import design_preemphasis_filter, design_underwater_filter, apply_filter, denoise_audio

"""
run_pipeline: Full Autotune Chain, Start to Finish
=====================================================

This function is the "spine" that connects every module you've built so
far, in the same order you've already tested them individually:

    load audio -> frame it -> detect pitch per frame -> find nearest
    scale note per frame -> compute how much to shift each frame ->
    shift (naive or phase vocoder) -> stitch back together (overlap-add)

Nothing new is invented here - this function just calls, in order, the
functions you already built and verified separately.
"""

def run_pipeline(input_path, config, use_phase_vocoder=True, use_preemphasis=True,
                 use_formant_preservation=True, use_underwater=False,
                 use_noise_cancellation=True, preserve_vibrato=True,
                 onset_protection_ms=30.0, humanize_cents=0.0):
    """
    input_path: str - path to a WAV file to correct
    config: AutoTuneConfig - holds frame_size, hop_size, sample_rate,
            scale_root, scale_type, correction_strength
    """

    raw_audio, sr = load_audio(input_path, target_sr=config.sample_rate)

    # Noise cancellation: Spectral subtraction + adaptive noise gate
    # Removes background room noise, mic hiss, fan hum, and outside noises before pitch detection
    if use_noise_cancellation:
        raw_audio = denoise_audio(raw_audio, sr)

    # Pre-emphasis filter: boosts high frequencies before pitch detection,
    # since voice naturally has weaker high-frequency energy.
    # We use detection_audio for pitch detection, but keep raw_audio for
    # pitch shifting so output volume and low-frequency vocal body are fully preserved.
    if use_preemphasis:
        b, a = design_preemphasis_filter(coeff=0.95)
        detection_audio = apply_filter(raw_audio, b, a)
    else:
        detection_audio = raw_audio

    # Framing: raw_audio for output synthesis, detection_audio for pitch detection
    frames, pad_len = frame_signal(raw_audio, config)
    detection_frames, _ = frame_signal(detection_audio, config)

    detected_pitches = detect_pitch_for_all_frames(detection_frames, sr)

    scale = build_scale_midi_set(config.scale_root, config.scale_type)

    # For each frame: if it's voiced (pitch > 0), find its nearest scale
    # note. If it's silent/unvoiced, target stays 0 (compute_shift_ratios
    # already knows to leave unvoiced frames un-shifted).
    target_pitches = np.zeros_like(detected_pitches)
    for i in range(len(detected_pitches)):
        if detected_pitches[i] > 0:
            target_pitches[i] = nearest_scale_note(detected_pitches[i], scale)

    shift_ratios = compute_shift_ratios(
        detected_pitches, target_pitches,
        strength=config.correction_strength,
        preserve_vibrato=preserve_vibrato,
        onset_protection_ms=onset_protection_ms,
        humanize_cents=humanize_cents,
        hop_size=config.hop_size,
        sample_rate=config.sample_rate
    )

    shift_ratios = smooth_shift_ratios(shift_ratios, detected_pitches, config.hop_size, config.sample_rate, config.retune_ms)

    if use_phase_vocoder:
        shifted_frames = phase_vocoder_shift(frames, shift_ratios, config)
        if use_formant_preservation:
            from .phase_vocoder import formant_preserve
            for i in range(len(frames)):
                shifted_frames[i] = formant_preserve(shifted_frames[i], frames[i], config)
    else:
        shifted_frames = np.zeros_like(frames)
        for i in range(len(frames)):
            shifted_frames[i] = naive_pitch_shift(frames[i], shift_ratios[i])

    corrected_audio = overlap_add(shifted_frames, config, pad_len)

    # Optional Underwater / Sub-aquatic Low-Pass Filter effect (cutoff at 800 Hz)
    if use_underwater:
        b_uw, a_uw = design_underwater_filter(cutoff=800, sample_rate=sr)
        corrected_audio = apply_filter(corrected_audio, b_uw, a_uw)

    # Match RMS perceived energy to raw input audio AFTER all filters so perceived volume never drops
    rms_raw = np.sqrt(np.mean(raw_audio ** 2))
    rms_corr = np.sqrt(np.mean(corrected_audio ** 2))
    if rms_corr > 1e-6 and rms_raw > 1e-6:
        gain = rms_raw / rms_corr
        corrected_audio = corrected_audio * gain
        # Safety peak clamp to avoid clipping distortion if gain boost occurs
        max_val = np.max(np.abs(corrected_audio))
        if max_val > 0.95:
            corrected_audio = corrected_audio * (0.95 / max_val)

    return {
        "raw_audio": raw_audio,
        "original_audio": raw_audio,
        "corrected_audio": corrected_audio,
        "sample_rate": sr,
        "detected_pitches": detected_pitches,
        "target_pitches": target_pitches,
        "shift_ratios": shift_ratios,
    }