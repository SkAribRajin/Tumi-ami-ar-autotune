# Retune speed and naturalness: one parameter, one code path

## The control

| Parameter | Range | Role |
|---|---|---|
| `retune_ms` | 0 – 200 ms | **Primary axis.** Moves the output continuously from robotic (0) to natural (50–120) to barely corrected (200). |
| `correction_strength` | 0 – 1 | Secondary blend. Scales how much of the correction is applied. Below 1 the singer is left partly out of tune; it doesn't make the sound more human the way retune does. |

Both are in the UI. The retune slider is now labelled "robotic ← → natural"
and runs to 200 ms.

## The one mechanism

Everything is computed in **semitones**, per frame `i`
(`pitch_shift.compute_shift_ratios`):

```
error[i]      = 12·log2(target[i] / detected[i])       how far this frame is from its note (clamped to ±2)
u[i]          = confidence[i] · error[i]               confidence gating
c[i]          = c[i-1] + α·(u[i] − c[i-1])             one-pole low-pass, α = 1 − exp(−hop/τ), τ = retune_ms
ratio[i]      = 2^(strength · c[i] / 12)
```

`c` starts at 0 and decays toward 0 through unvoiced gaps.

### Why one low-pass filter produces both "robotic" and "melodious"

Write the sung pitch as its note plus a deviation: `p(t) = note + d(t)`.
Here `d` contains everything the singer does away from the note: a steady
tuning error (slow), scoops into notes (~50–150 ms), and vibrato (~4.5–7 Hz).
The target is the note, so `error = note − p = −d`.

The output pitch is

```
out = p + LPF(error) = note + d − LPF(d) = note + HPF(d)
```

So the output is the note plus a **high-passed** copy of what the singer did.
A one-pole low-pass with time constant τ has its corner at `fc = 1/(2πτ)`,
and its complement passes everything above `fc`:

- deviations **slower** than `fc` (a note held 30 cents flat, slow drift) are
  removed, so those notes end up in tune
- deviations **faster** than `fc` (vibrato, scoops, onsets) pass through
  unchanged, so they sound human

Moving τ slides `fc` across the frequencies that make a voice sound human:

| `retune_ms` | `fc` | Vibrato kept at 5.5 Hz (theory, discrete filter) | Measured on synthetic ballad | Character |
|---|---|---|---|---|
| 0 | ∞ (no filter) | 0% | 24%* | Hard snap. Vibrato flattened, note changes are steps: the classic robotic sound |
| 10 | 15.9 Hz | 17% | 38%* | Very tight, audibly processed |
| 25 | 6.4 Hz | 51% | 60% | Tight modern pop |
| 50 | 3.2 Hz | 77% | 80% | Natural; notes land in tune, vibrato mostly intact |
| 100 | 1.6 Hz | 91% | 91% | Transparent; scoops audible |
| 200 | 0.8 Hz | 96% | 96% | Only slow drift corrected |

\* The measurement floor is about 5 cents of residual pitch noise (24% of the
21-cent input vibrato), so small values read high. The "robotic" end is
already the case at 0 ms: the median distance from the note there is
**1.4 cents** (input: 32 cents).

The note centre is corrected at every setting. What τ changes is how long it
takes (about τ to cover 63% of the error) and how much of the fast movement
survives. That is why one continuous parameter covers both ends: "robotic"
and "melodious" are not two algorithms. They are two corner frequencies of
the same filter, one above the vibrato rate and one below it.

## How the other "naturalness" features fall out of the same filter

| Feature | Old implementation | Now |
|---|---|---|
| **Vibrato preservation** | A separate 200 ms rolling mean in Hz. It averaged octave errors (±10 st shifts) and blended notes at transitions, and with `preserve_vibrato=True` (the hidden default) it kept vibrato **even at retune 0**, so the robotic end was unreachable | It is the high-pass complement of the retune filter. Vibrato is kept in proportion to how far 5.5 Hz lies above `fc` (table above) |
| **Note-onset protection** | A separate strength ramp over 30 ms, then undone by `smooth_shift_ratios`, which restarted each voiced run at the **full** unsmoothed ratio: an instant snap on every syllable | It is the filter's starting state. `c` begins at 0, so for the first ~τ of a note after a gap the attack is heard as sung while the correction ramps in. At τ = 0 there is no protection (robotic); at τ = 100 ms a 60 ms scoop passes almost intact |
| **Confidence gating** | Hard voiced/unvoiced threshold (autocorrelation peak > 0.3). Noise was shifted and breathy frames toggled | Soft weight on the filter **input**. A breathy frame pushes the correction a little. An unvoiced frame pushes nothing, so the correction decays smoothly instead of switching off and on |
| **Glide between notes** | Implicit in the EMA, but reset at each gap | When the target note changes, `error` steps, and the low-pass turns the step into a glide of time constant τ: a hard step at 0, a smooth portamento at 50+ |

Nothing here branches on "mode". Each feature is a consequence of `α`
(from τ), the filter's initial state, or the weight on its input.

## Things that used to make it sound robotic *regardless* of the parameter

These were fixed first. Without those fixes no retune setting could sound
natural (see `diagnosis.md`):

1. **Phase vocoder incoherence.** Even at ratio 1.0 the output was
   phase-scrambled, so every frame was degraded. It is now an exact identity
   at ratio 1.0.
2. **Pitch-detector jitter.** Integer lags meant 17–31 cent steps from frame
   to frame. With `retune_ms = 0` the output pitch is
   `note + (true pitch − detected pitch)`, so detector error goes straight to
   the output. YIN with parabolic interpolation is accurate to under 3 cents
   (tested from 73 to 880 Hz).
3. **Target flip-flop.** Per-frame nearest note, fixed by
   `scales.quantize_pitch_track`:
   - note **centre** = 350 ms running median (about two vibrato cycles; the
     median over whole cycles is the vibrato centre, and a median, unlike a
     mean, keeps note steps sharp)
   - **hysteresis** 0.25 semitones on the centre
   - **fast-note override**: if the short-term (58 ms median) pitch stays more
     than 0.85 semitones from the current note for 58 ms, switch and backfill.
     A vibrato peak can poke past 0.85 but doesn't stay there.

   Measured over 4.5–7 Hz vibrato: at most one note change (a single settle,
   never back-and-forth) for depths of 20–50 cents centred 35 cents off a
   note, and for depths up to 40 cents centred exactly on the midpoint. The
   old per-frame choice flipped on every vibrato cycle in these cases. (A ±50-cent vibrato centred on the midpoint touches both
   notes equally and is genuinely ambiguous.) Notes down to 70 ms long are
   followed without lag.

## Recommended settings

| Goal | `retune_ms` | `correction_strength` |
|---|---|---|
| Hard-tuned effect (T-Pain / melodic-rap style) | 0–5 | 1.0 |
| Polished pop | 20–35 | 1.0 |
| Natural ballad correction | 60–120 | 0.8–1.0 |
| Light touch / already well sung | 150–200 | 1.0 |

The default is 40 ms / 1.0.
