# this was created Week 5-6: full pipeline + customization features (scale selector, correction strength)
# to run the full autotune pipeline from start to finish, using the modules i built in previous weeks.

import os
import sys
sys.path.insert(0, '.')
from src.autotune.config import AutoTuneConfig
from src.autotune.pipeline import run_pipeline
from src.autotune.io_utils import save_audio
from src.autotune.visualization import (
    plot_waveform_comparison,
    plot_spectrogram_comparison,
    plot_pitch_contour,
)
from src.autotune.filters import (
    design_preemphasis_filter,
    plot_pole_zero,
    plot_frequency_response,
)

"""
run_demo.py: Simple Entry Point (edit variables below, then run)
====================================================================

No command-line flags - just change these variables directly each time
you want to try a different setting, then run:
    python run_demo.py
"""

# ---- EDIT THESE SETTINGS ----
INPUT_FILE = "data/raw/test_voice.wav"
OUTPUT_FILE = "data/processed/output.wav"
PLOTS_DIR = "data/processed/plots"
SCALE_ROOT = "C"              # e.g. "C", "A", "G"
SCALE_TYPE = "major"          # "major", "natural_minor", or "chromatic"
CORRECTION_STRENGTH = 1.0     # 0.0 = no correction, 1.0 = full correction
USE_PHASE_VOCODER = True      # False = use naive resampling shift instead
# ------------------------------

os.makedirs(PLOTS_DIR, exist_ok=True)

config = AutoTuneConfig(scale_root=SCALE_ROOT,
                         scale_type=SCALE_TYPE,
                         correction_strength=CORRECTION_STRENGTH)

result = run_pipeline(INPUT_FILE, config, use_phase_vocoder=USE_PHASE_VOCODER)
save_audio(OUTPUT_FILE, result["corrected_audio"], result["sample_rate"])

print(f"Saved corrected audio to {OUTPUT_FILE}")
print(f"Settings: scale={SCALE_ROOT} {SCALE_TYPE}, strength={CORRECTION_STRENGTH}, "
      f"method={'phase vocoder' if USE_PHASE_VOCODER else 'naive'}")

# Generate visual comparisons and plots
plot_waveform_comparison(
    result["raw_audio"], result["corrected_audio"],
    result["sample_rate"], os.path.join(PLOTS_DIR, "waveform_comparison.png")
)
plot_spectrogram_comparison(
    result["raw_audio"], result["corrected_audio"],
    result["sample_rate"], os.path.join(PLOTS_DIR, "spectrogram_comparison.png")
)
plot_pitch_contour(
    result["detected_pitches"], result["target_pitches"],
    result["detected_pitches"] * result["shift_ratios"], config.hop_size,
    result["sample_rate"], os.path.join(PLOTS_DIR, "pitch_contour.png")
)

b, a = design_preemphasis_filter(coeff=0.95)
plot_pole_zero(b, a, os.path.join(PLOTS_DIR, "filter_pole_zero.png"))
plot_frequency_response(b, a, result["sample_rate"], os.path.join(PLOTS_DIR, "filter_freq_response.png"))

print(f"Generated comparison plots in {PLOTS_DIR}/")




'''
Koyekta Custom Test             python run_demo.py

   OUTPUT_FILE = "data/processed/demo_full_strength.wav"
   CORRECTION_STRENGTH = 1.0
   USE_PHASE_VOCODER = True



    OUTPUT_FILE = "data/processed/demo_half_strength.wav"
    CORRECTION_STRENGTH = 0.5



    OUTPUT_FILE = "data/processed/demo_naive.wav"
    USE_PHASE_VOCODER = False



'''