import os
import sys
import numpy as np

sys.path.insert(0, '.')

from src.autotune.config import AutoTuneConfig
from src.autotune.pipeline import run_pipeline
from src.autotune.io_utils import save_audio, load_audio
from src.autotune.framing import frame_signal
from src.autotune.pitch_detection import detect_pitch_for_all_frames
from src.autotune.filters import (
    design_preemphasis_filter,
    apply_filter,
    plot_pole_zero,
    plot_frequency_response,
)
from src.autotune.visualization import (
    plot_waveform_comparison,
    plot_spectrogram_comparison,
    plot_pitch_contour,
)
import matplotlib.pyplot as plt


def generate_supervisor_demo_suite():
    """
    Generates a complete suite of evaluation audio clips and high-resolution DSP plots
    specifically designed to demonstrate the core DSP engineering thesis to a supervisor:

    1. A/B/C Playback (Original -> Naive Resampling -> Phase Vocoder)
    2. Correction Strength Sweep (0.0 -> 0.5 -> 1.0)
    3. Retune Speed Demo (0ms T-Pain snap -> 40ms Natural -> 100ms Slow glide)
    4. Pre-emphasis On/Off Pitch Detection Contour Comparison
    5. Spectrogram Harmonic Snapping & Pitch Contour Overlay Plots
    """
    print("🚀 Generating Supervisor Evaluation Demo Suite...")

    input_file = "data/raw/test_voice.wav"
    output_dir = "data/processed/supervisor_demo"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Ensure test voice exists
    if not os.path.exists(input_file):
        import gen_test_tone

    config = AutoTuneConfig(scale_root="C", scale_type="major", correction_strength=1.0)

    # 1. A/B/C Playback: Original -> Naive -> Phase Vocoder
    print("1/5 Generating A/B/C Comparison Audio (Original -> Naive -> Phase Vocoder)...")
    res_phase = run_pipeline(input_file, config, use_phase_vocoder=True)
    res_naive = run_pipeline(input_file, config, use_phase_vocoder=False)

    sr = res_phase["sample_rate"]
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)

    abc_audio = np.concatenate([
        res_phase["raw_audio"],
        silence,
        res_naive["corrected_audio"],
        silence,
        res_phase["corrected_audio"]
    ])
    save_audio(os.path.join(output_dir, "A_B_C_comparison.wav"), abc_audio, sr)

    # 2. Correction Strength Sweep (0.0 -> 0.5 -> 1.0)
    print("2/5 Generating Correction Strength Sweep (0.0 -> 0.5 -> 1.0)...")
    cfg_0 = AutoTuneConfig(scale_root="C", scale_type="major", correction_strength=0.0)
    cfg_50 = AutoTuneConfig(scale_root="C", scale_type="major", correction_strength=0.5)
    cfg_100 = AutoTuneConfig(scale_root="C", scale_type="major", correction_strength=1.0)

    res_0 = run_pipeline(input_file, cfg_0)
    res_50 = run_pipeline(input_file, cfg_50)
    res_100 = run_pipeline(input_file, cfg_100)

    strength_sweep = np.concatenate([
        res_0["corrected_audio"],
        silence,
        res_50["corrected_audio"],
        silence,
        res_100["corrected_audio"]
    ])
    save_audio(os.path.join(output_dir, "strength_sweep_0_50_100.wav"), strength_sweep, sr)

    # 3. Retune Speed Sweep (0ms T-Pain snap -> 40ms Natural -> 100ms Slow glide)
    print("3/5 Generating Retune Speed Sweep (0ms T-Pain -> 40ms Natural -> 100ms Slow)...")
    cfg_tpain = AutoTuneConfig(scale_root="C", scale_type="major", retune_ms=0.0)
    cfg_natural = AutoTuneConfig(scale_root="C", scale_type="major", retune_ms=40.0)
    cfg_slow = AutoTuneConfig(scale_root="C", scale_type="major", retune_ms=100.0)

    res_tpain = run_pipeline(input_file, cfg_tpain)
    res_nat = run_pipeline(input_file, cfg_natural)
    res_slow = run_pipeline(input_file, cfg_slow)

    retune_sweep = np.concatenate([
        res_tpain["corrected_audio"],
        silence,
        res_nat["corrected_audio"],
        silence,
        res_slow["corrected_audio"]
    ])
    save_audio(os.path.join(output_dir, "retune_speed_tpain_vs_natural.wav"), retune_sweep, sr)

    # 4. Pre-emphasis On/Off Pitch Detection Comparison Plot
    print("4/5 Generating Pre-emphasis On/Off Pitch Contour Comparison Plot...")
    raw_audio, sr = load_audio(input_file, target_sr=config.sample_rate)
    b, a = design_preemphasis_filter(coeff=0.95)
    filtered_audio = apply_filter(raw_audio, b, a)

    raw_frames, _ = frame_signal(raw_audio, config)
    filt_frames, _ = frame_signal(filtered_audio, config)

    pitch_raw = detect_pitch_for_all_frames(raw_frames, sr)
    pitch_filt = detect_pitch_for_all_frames(filt_frames, sr)

    t_axis = np.arange(len(pitch_raw)) * config.hop_size / float(sr)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_axis, np.where(pitch_raw > 0, pitch_raw, np.nan), label="Raw Audio Pitch", color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.plot(t_axis, np.where(pitch_filt > 0, pitch_filt, np.nan), label="Pre-emphasized Pitch (Cleaner Tracking)", color="#2ecc71", linewidth=1.8)
    ax.set_title("Pre-emphasis Filter Impact on Pitch Detection Stability")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "preemphasis_pitch_stability.png"))
    plt.close(fig)

    # 5. Spectrogram & Pitch Contour Overlay Plots
    print("5/5 Generating Spectrogram and Pitch Contour Overlay Plots...")
    plot_waveform_comparison(
        res_phase["raw_audio"], res_phase["corrected_audio"],
        sr, os.path.join(plots_dir, "waveform_comparison.png")
    )
    plot_spectrogram_comparison(
        res_phase["raw_audio"], res_phase["corrected_audio"],
        sr, os.path.join(plots_dir, "spectrogram_comparison.png")
    )
    plot_pitch_contour(
        res_phase["detected_pitches"], res_phase["target_pitches"],
        res_phase["detected_pitches"] * res_phase["shift_ratios"],
        config.hop_size, sr, os.path.join(plots_dir, "pitch_contour.png")
    )
    plot_pole_zero(b, a, os.path.join(plots_dir, "filter_pole_zero.png"))
    plot_frequency_response(b, a, sr, os.path.join(plots_dir, "filter_freq_response.png"))

    print(f"\n✅ Supervisor Evaluation Demo Suite successfully created in:\n   📁 {output_dir}/")
    print(f"   📁 Audio clips: A_B_C_comparison.wav, strength_sweep_0_50_100.wav, retune_speed_tpain_vs_natural.wav")
    print(f"   📁 High-res plots saved to: {plots_dir}/")


if __name__ == "__main__":
    generate_supervisor_demo_suite()
