import numpy as np
from scipy.ndimage import median_filter

"""
A bit of music theory:
    frequency of A4 = 440Hz, A5 = 880Hz
    880 = 440.2; so one octave corresponds to doubling the frequency.
    But musical notes dont increase by a fixed number of Hz. So musical pitch is
    used in the logarithmic scale. MIDI(Musical Instrument Digital Interface( numbers give us a convenient representation. 

    ex: C4 -> 60
        C#$ -> 61 and so on
    one semi tone = 1 MIDI number
    one octave = 12 MIDI numbers
    MIDI has assigned A4 = 69 and at standard tuning A4=440Hz so we use this as a reference point

    f = 440*2^((n-69)/12); n = MIDI note number
    ex: A3 is one octave below A4 so n=69-12=57
    f = 440*2^(-12/12) = 220Hz
    A#4 is one semitone above A4

    f=440*2^((70-69)/12)=466.26Hz

    But our pitch detector gives f
    so n = 69+12log(f/440)
    MIDI number is not always an integer. If MIDI = 69.39 it means the detected pitch is about 39%
    of the way from MIDI note 69 to MIDI note 70
    rounding gives us the nearest chromatic note. Nearest integer to 69.39 is 69 so A4. If n=69,8 then
    rounding gives 70 which is A#4

    *This has a pretty big caveat*
    Suppose we choose C major scale, C major contains: C,D,E,F,G,A,B

"""

A4_FREQ = 440.0
A4_MIDI = 69

SCALE_INTERVALS = {
    "major": [0,2,4,5,7,9,11],
    "natural_minor": [0,2,3,5,7,8,10],
    "chromatic": list(range(12))
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_name_to_midi(note_name, octave=4):
    idx = NOTE_NAMES.index(note_name)
    return (octave+1)*12 +idx

def freq_to_midi(frequency):
    if frequency <= 0:
        return None
    return A4_MIDI + 12*np.log2(frequency/A4_FREQ)

def midi_to_freq(midi_note):
    return A4_FREQ*(2**((midi_note - A4_MIDI)/12))

def build_scale_midi_set(root_note_name, scale_type, midi_low=24, midi_high=108):
    root_pitch_class = NOTE_NAMES.index(root_note_name)
    intervals = SCALE_INTERVALS[scale_type]

    allowed = []
    for midi in range(midi_low, midi_high+1):
        pitch_class = midi%12
        offset = (pitch_class - root_pitch_class)%12
        if offset in intervals:
            allowed.append(midi)
    return np.array(allowed)

def nearest_scale_note(frequency, scale_midi_set):
    midi = freq_to_midi(frequency)
    if midi is None:
        return 0.0
    idx = np.argmin(np.abs(scale_midi_set - midi))
    nearest_midi = scale_midi_set[idx]

    return midi_to_freq(nearest_midi)


def quantize_pitch_track(pitches, scale_midi_set, hop_size=512, sample_rate=44100,
                         hysteresis=0.25, center_ms=350.0, jump_semitones=0.85,
                         confirm_ms=58.0):
    """
    Track-level version of nearest_scale_note, used by the pipeline.

    Calling nearest_scale_note independently per frame has a failure mode:
    a singer holding a note near the midpoint between two scale notes (or
    with vibrato wider than the distance to that midpoint) makes the target
    flip between the two notes from frame to frame. That is a +/-1-2
    semitone square-wave warble in the output, and it sounds robotic at ANY
    correction strength.

    Within each voiced run (the choice resets at every unvoiced gap):
      1. Note CENTRE = running median of the pitch (semitones) over
         center_ms, about two vibrato cycles (4-7 Hz -> 140-250 ms per cycle).
         The median over whole cycles is the vibrato centre; over ~1.5 cycles
         a residual of +/-14 cents remained, enough to cross the hysteresis.
         A running MEAN would
         also smear fast note steps; that kept the old target for ~50 ms
         after each rap syllable changed note (measured -135 cent
         corrections on a line that was only 35 cents sharp).
      2. Hysteresis: switch to a neighbouring note only when the centre is
         closer to it by more than `hysteresis` semitones.
      3. Fast notes: a long median would swallow notes shorter than half its
         window (fast rap/melisma). So if the short-term pitch (58 ms median)
         stays more than `jump_semitones` from the current note for
         confirm_ms, it's a new note: switch, and backfill the confirming
         frames. A vibrato peak can poke past jump_semitones but doesn't stay
         there that long.

    Returns target frequencies in Hz, shape (num_frames,), 0.0 where unvoiced.
    """
    pitches = np.asarray(pitches, dtype=np.float64)
    n = len(pitches)
    targets = np.zeros(n)
    voiced = pitches > 0
    if not voiced.any():
        return targets

    midi = np.zeros(n)
    midi[voiced] = A4_MIDI + 12 * np.log2(pitches[voiced] / A4_FREQ)
    frames_per_ms = sample_rate / hop_size / 1000.0
    long_size = 2 * max(1, int(round(center_ms * frames_per_ms / 2))) + 1
    short_size = 2 * max(1, int(round(58.0 * frames_per_ms / 2))) + 1
    confirm_frames = max(1, int(round(confirm_ms * frames_per_ms)))
    scale = np.asarray(scale_midi_set, dtype=np.float64)
    nearest = lambda m: scale[np.argmin(np.abs(scale - m))]

    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        run = midi[i:j]
        centre = median_filter(run, size=long_size, mode='nearest')
        short = median_filter(run, size=short_size, mode='nearest')

        current = nearest(centre[0])
        away = 0  # consecutive frames the short-term pitch has been far from `current`
        for t in range(len(run)):
            if abs(short[t] - current) > jump_semitones:
                away += 1
                if away >= confirm_frames:
                    # a real new note, not a vibrato peak: switch, and backfill
                    # the frames spent confirming it (we're offline, no latency)
                    current = nearest(short[t])
                    targets[i + t - away + 1:i + t] = midi_to_freq(current)
                    away = 0
            else:
                away = 0
                candidate = nearest(centre[t])
                # the short-term pitch must agree, otherwise the long centre
                # would undo a fast note it has swallowed
                if (candidate != current
                        and abs(centre[t] - current) - abs(centre[t] - candidate) > hysteresis
                        and abs(short[t] - candidate) < abs(short[t] - current)):
                    current = candidate
            targets[i + t] = midi_to_freq(current)
        i = j

    return targets
