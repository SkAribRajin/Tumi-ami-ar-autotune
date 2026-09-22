# this was created  Week 5-6: full pipeline + customization features (scale selector, correction strength)


import numpy as np
from .io_utils import load_audio, save_audio
from .framing import frame_signal, overlap_add
from .pitch_detection import detect_pitch_track
from .scales import build_scale_midi_set, quantize_pitch_track
from .pitch_shift import compute_shift_ratios, naive_pitch_shift
from .phase_vocoder import phase_vocoder_shift
from .filters import (design_preemphasis_filter, design_underwater_filter, apply_filter,
                      denoise_audio, soft_limit)

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

def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if len(x) else 0.0


def run_pipeline(input_path, config, use_phase_vocoder=True, use_preemphasis=False,
                 use_formant_preservation=True, use_underwater=False,
                 use_noise_cancellation=True, humanize_cents=0.0):
    """
    input_path: str - path to a WAV file to correct
    config: AutoTuneConfig - frame/hop/sample rate, scale, correction_strength,
            retune_ms (the natural <-> robotic control), pitch range, hysteresis

    Signal path and expected shape/scale at each step: docs/signal-routing.md
    """

    # float32 mono in [-1, 1], resampled to config.sample_rate
    raw_audio, sr = load_audio(input_path, target_sr=config.sample_rate)

    # Stationary-noise reduction. Returns the input unchanged for clean stems.
    if use_noise_cancellation:
        raw_audio = denoise_audio(raw_audio, sr)

    # Pre-emphasis only feeds pitch DETECTION; synthesis always uses raw_audio.
    # Off by default: it tilts energy toward upper harmonics, which is what
    # makes a period detector choose a harmonic instead of the fundamental.
    if use_preemphasis:
        b, a = design_preemphasis_filter(coeff=0.95)
        detection_audio = apply_filter(raw_audio, b, a)
    else:
        detection_audio = raw_audio

    # Hann-windowed frames for synthesis; detection frames them itself
    # (same centres, no window).
    frames, pad_len = frame_signal(raw_audio, config)
    detected_pitches, confidence = detect_pitch_track(detection_audio, config)

    scale = build_scale_midi_set(config.scale_root, config.scale_type)
    target_pitches = quantize_pitch_track(detected_pitches, scale, config.hop_size,
                                          config.sample_rate, config.note_hysteresis)

    shift_ratios = compute_shift_ratios(
        detected_pitches, target_pitches,
        strength=config.correction_strength,
        retune_ms=config.retune_ms,
        confidence=confidence,
        hop_size=config.hop_size,
        sample_rate=config.sample_rate,
        humanize_cents=humanize_cents,
    )

    if use_phase_vocoder:
        shifted_frames = phase_vocoder_shift(frames, shift_ratios, config,
                                             preserve_formants=use_formant_preservation)
    else:
        shifted_frames = np.zeros_like(frames)
        for i in range(len(frames)):
            shifted_frames[i] = naive_pitch_shift(frames[i], shift_ratios[i])

    corrected_audio = overlap_add(shifted_frames, config, pad_len, length=len(raw_audio))

    if use_underwater:
        b_uw, a_uw = design_underwater_filter(cutoff=800, sample_rate=sr)
        corrected_audio = apply_filter(corrected_audio, b_uw, a_uw)

    # Level: the phase vocoder preserves energy, so for pitch correction this
    # gain is ~1.0 (it is reported in the result so a regression is visible).
    # It is bounded to +/-6 dB so it can never be used to hide a broken stage
    # the way the old unbounded RMS match did. Then a peak limiter, NOT a
    # whole-file rescale, keeps the result below full scale.
    rms_in, rms_out = _rms(raw_audio), _rms(corrected_audio)
    makeup_gain = float(np.clip(rms_in / rms_out, 0.5, 2.0)) if rms_in > 1e-7 and rms_out > 1e-7 else 1.0
    corrected_audio = soft_limit(corrected_audio * makeup_gain, sr)

    return {
        "raw_audio": raw_audio,
        "original_audio": raw_audio,
        "corrected_audio": corrected_audio,
        "sample_rate": sr,
        "detected_pitches": detected_pitches,
        "target_pitches": target_pitches,
        "shift_ratios": shift_ratios,
        "confidence": confidence,
        "makeup_gain": makeup_gain,
    }
