import numpy as np

"""
Discrete Fourier Transform:
X[k] = ∑n=0 to N-1 x[n]e^(-j2pikn/N

Continuous Fourier Transform:
X(f) = ∫x(t)e^(-j2pift)dt

In python we find the integration using trapezoidal rule which basically converts CFT
calculation to DFT calculation. FFT(Fast Fourier Transform) is an algorithm that calculates
the DFT much faster. it takes O(NlogN) time instead O(N^2)

"""

def autocorrelate(frame):
    """
        1. Autocorrelation measure how much a signal resembles a shifted/delayed version of itself.
        For pitch detection we are basically asking if we move the audio waveform to the right then
        at what distance does it line up with itself again.

        suppose for x[n] the period ,T=5 samples. if the audio sampling rate is 1000 samples/sec:
        then f=sampling rate/time period in samples= 1000/5 = 200Hz

        x = [1 0 -1 0 1 0 -1 0] there is a pattern
        if we shift the array by 4 then the pattern lines up again perfectly. so autocorrelation tests
        the lag where we get a strong match. [lag = number of samples we shift]

        The Autocorrelation Formula:

        R(k) = ∑n=0 to N-k-1 x[n]x[n+k]
        let k = 4
        then we are comparing
        x[n]:   1 0 -1 0
        x[n+4]: 1 0 -1 0
        We multiply corresponding values: 1*1+(-1)(-1) = 2
        This is a large number beacuse the two portions are very similar
        If the lag does not match, suppose k =1
        x[n]:   1 0 -1 0 1 0 -1
        x[n+4]: 0 -1 0 1 0 -1 0 ; the product is = 0

        Suppose a sung note has T=100 samples then autocorrelation will have a string 
        peak around k =100 so kpeak~100 and f0=fs/100=44100/100 = 441Hz, we estimate the singer's fundamental pitch as approx 441 Hz

        We cannot directly use FFT because if we just say take fpeak then other frequencies will be distorted.

        2. we exclude k=0 because at that point a signal will obviously be maximally similar to itself so we search for the next meaningful peak.


        3. We can narrow down the lag range for our search:
        fs=44100 and human pitch is approx: 80 Hz<= f<= 1000Hz
        we know k = fs/f so 44<= k <= 551

        4. Normalization Problem:
        R(k) =  ∑n=0 to N-k-1 x[n]x[n+k] for k=0: there are N terms. for k=100 there are N-100 terms
        so as k gets larger we are comparing fewer terms.
        Now suppose two lags produce exactly the same average similarity
        Suppose avg contributio is 5
        At lag 10, we compare 990 samples, R(10) = 990.5 = 4950
        at lag 500, we compare 500 samples R(500) = 500*5 = 2500
        so raw autocorrelation says R(10)>R(500) even tho the average similarity was identical. This is the
        Shrinking window problem. To fix this we divide by the no of overlapping samples:N-k

        Rnormalized(k) = R(k)/(N-k)
    """

    n = len(frame)
    #Zero pad to avoid circular correlation wraparound artifacts
    #For an N size sample we can shift from -(N-1) to N-1 so there are 2N-1 shifts possible
    #so we pad to the size of 2N
    padded_len = 2*n
    #rfft = real input fast fourier transform
    fft_frame = np.fft.rfft(frame, n = padded_len)
    power_spectrum = fft_frame*np.conj(fft_frame)
    result = np.fft.irfft(power_spectrum)

    #R[k] = IFFT(FFT(x).FFT(x)*) : Wiener–Khinchin theorem.

    return result[:n]

def detect_pitch_autocorrelation(frame, sample_rate, fmin=80, fmax = 1000):

    #T = fs/f example: fs = 44100 and fmax=1000 then Tmin = 44.1
    min_lag = int(sample_rate/fmax)
    max_lag = int(sample_rate/fmin)
    max_lag = min(max_lag, len(frame)-1)

    #we dont want to seach at lag 0 as R(0)=n∑​x[n]^2 we are comparing x[n] with itself
    if min_lag<1:
        min_lag = 1

    #corr[k]=R[k] where k=lag
    corr = autocorrelate(frame)

    #R(0) = ∑|x[n]|^2 s essentially the signal's energy, if the frame is [0, 0, 0, ..]
    #then there's isnt enough signal to determine a pitch
    if corr[0] <= 1e-8:
        return 0.0
    corr_normalized = corr/corr[0]

    search_region = corr_normalized[min_lag: max_lag]
    if len(search_region) == 0:
        return 0.0

    """
    search_region:

    position       value

    0              0.10
    1              0.20
    2              0.30
    3              0.80  ← largest
    4              0.40
    argmax return 3. 

    search_region[0] = corr_normalized[44]

    search_region[1] = corr_normalized[45]

    search_region[2] = corr_normalized[46]

    search_region[3] = corr_normalized[47]  

    so the actual peak lag is = 44+3 = 47=min lag+argmax val
    """
    peak_lag = np.argmax(search_region)+min_lag
    peak_value = corr_normalized[peak_lag]

    confidence_threshold = 0.3
    if peak_value < confidence_threshold:
        return 0.0
    frequency = sample_rate/peak_lag
    return frequency

"""
YIN pitch detection (de Cheveigne & Kawahara, 2002)
=====================================================
The plain autocorrelation detector above has three weaknesses that showed up
on real vocals (see docs/diagnosis.md):

  1. It picks an INTEGER lag with argmax. At 44.1 kHz a 440 Hz note has a
     period of ~100 samples, so one lag step is ~17 cents; at 800 Hz it is
     ~31 cents. The detected pitch jumps between neighbouring lags from frame
     to frame, and that jitter goes straight into the correction.
  2. corr/corr[0] on a Hann-windowed frame is biased toward short lags and
     makes octave errors (12.9% of voiced frames on my_voice.wav).
  3. Its "confidence" (peak > 0.3) lets breaths, "s"/"sh" and reverb tails
     through as voiced, so they get pitch shifted.

YIN fixes all three:
  d(tau)  = sum_j (x[j] - x[j+tau])^2                  difference function
  d'(tau) = d(tau) * tau / sum_{t=1..tau} d(t)         cumulative-mean normalised
  Pick the FIRST dip of d' below a threshold (not the global best, which is
  what causes octave-too-low errors), refine it with a parabola through the
  three points around the dip (sub-sample lag, ~1 cent accuracy), and report
  d'(tau) itself as the aperiodicity: ~0 for a clean sung vowel, ~1 for noise.
"""

def _yin_cmndf(frames, max_lag):
    """
    frames: (num_frames, N) UNWINDOWED frames. Returns d' with shape
    (num_frames, max_lag + 1). Computed for all frames at once via FFT:
        d(tau) = e(0) + e(tau) - 2 r(tau)
    where r is the cross-correlation of the first W samples with the frame,
    and e(tau) is the energy of the W samples starting at tau.
    """
    x = np.asarray(frames, dtype=np.float64)
    num_frames, n = x.shape
    w = n - max_lag
    nfft = 1 << int(np.ceil(np.log2(n + w)))
    r = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(x[:, :w], nfft)), nfft)[:, :max_lag + 1]
    csum = np.concatenate([np.zeros((num_frames, 1)), np.cumsum(x * x, axis=1)], axis=1)
    lags = np.arange(max_lag + 1)
    e0 = csum[:, w][:, None]
    e_tau = csum[:, lags + w] - csum[:, lags]
    d = np.maximum(e0 + e_tau - 2.0 * r, 0.0)
    d[:, 0] = 0.0
    cum = np.cumsum(d[:, 1:], axis=1)
    cmndf = np.ones_like(d)
    cmndf[:, 1:] = d[:, 1:] * lags[1:] / np.maximum(cum, 1e-12)
    # digital silence: d == 0 everywhere would read as "perfectly periodic"
    silent = cum[:, -1] <= 1e-12 * max_lag
    cmndf[silent] = 1.0
    return cmndf


def detect_pitch_yin(frames, sample_rate, fmin=60.0, fmax=1100.0, threshold=0.15, chunk=256):
    """
    frames: (num_frames, N) UNWINDOWED frames (see frame_signal(apply_window=False)).

    Returns (pitches_hz, aperiodicity), both shape (num_frames,).
    aperiodicity is d'(best lag): 0 = perfectly periodic, 1 = noise. A pitch
    is returned for every frame; the caller decides what counts as voiced.
    """
    frames = np.atleast_2d(frames)
    n = frames.shape[1]
    min_lag = max(2, int(np.floor(sample_rate / fmax)))
    max_lag = min(int(np.ceil(sample_rate / fmin)), n // 2)
    pitches = np.zeros(frames.shape[0])
    aperiodicity = np.ones(frames.shape[0])

    for c0 in range(0, frames.shape[0], chunk):
        cm = _yin_cmndf(frames[c0:c0 + chunk], max_lag)
        region = cm[:, min_lag:max_lag]
        # local minima of d' inside the search range
        local_min = np.zeros_like(region, dtype=bool)
        local_min[:, 1:-1] = (region[:, 1:-1] <= region[:, :-2]) & (region[:, 1:-1] <= region[:, 2:])
        good = local_min & (region < threshold)
        # nothing under the threshold (breathy/noisy frame): take the FIRST
        # dip that is nearly as deep as the deepest one, not the deepest one
        # itself, which is often the dip at twice the period (octave-low error)
        near_best = local_min & (region <= region.min(axis=1, keepdims=True) + 0.1)
        fallback = np.where(near_best.any(axis=1), np.argmax(near_best, axis=1), np.argmin(region, axis=1))
        idx = np.where(good.any(axis=1), np.argmax(good, axis=1), fallback)
        tau = idx + min_lag

        # parabolic interpolation around the dip -> fractional lag
        rows = np.arange(len(tau))
        a = cm[rows, tau - 1]
        b = cm[rows, tau]
        c = cm[rows, np.minimum(tau + 1, max_lag)]
        denom = a - 2.0 * b + c
        safe = np.abs(denom) > 1e-12
        shift = np.zeros_like(b)
        shift[safe] = 0.5 * (a[safe] - c[safe]) / denom[safe]
        shift = np.clip(shift, -1.0, 1.0)

        pitches[c0:c0 + chunk] = sample_rate / (tau + shift)
        aperiodicity[c0:c0 + chunk] = np.clip(b, 0.0, 1.0)

    return pitches, aperiodicity


def voicing_confidence(aperiodicity, frame_rms, voiced_below=0.2, unvoiced_above=0.45,
                       silence_db=-50.0):
    """
    Turns YIN aperiodicity + frame loudness into a soft 0..1 weight.

    1.0 below `voiced_below`, 0.0 above `unvoiced_above`, linear in between.
    A soft weight (not a hard voiced/unvoiced switch) matters: the correction
    amount is multiplied by it, so a breathy frame gets a little correction
    rather than toggling between "full shift" and "no shift" frame to frame.

    The loudness gate is RELATIVE to the loud part of the track (95th
    percentile frame RMS), so it behaves the same on a quiet phone recording
    and on a mastered, compressed rap vocal.
    """
    conf = np.clip((unvoiced_above - aperiodicity) / (unvoiced_above - voiced_below), 0.0, 1.0)
    rms_db = 20 * np.log10(np.asarray(frame_rms) + 1e-12)
    loud_ref = np.percentile(rms_db, 95) if len(rms_db) else 0.0
    conf[rms_db < loud_ref + silence_db] = 0.0
    conf[rms_db < -90.0] = 0.0
    return conf


def clean_pitch_track(pitches, confidence, octave_window=12, max_gap=2):
    """
    Post-processing on the voiced frames (confidence > 0), all in log pitch:
      1. Octave-error folding: a frame more than 7 semitones from the running
         median of its neighbourhood is moved by whole octaves toward it.
      2. 3-point median within voiced runs removes single-frame outliers.
      3. Dropouts of <= max_gap frames inside a note are filled by
         interpolation (with half the neighbours' confidence) so the
         correction filter doesn't see the note stop and restart.
    """
    confidence = confidence.copy()
    voiced = confidence > 0
    n = len(pitches)
    if not voiced.any():
        return np.zeros(n), confidence

    semis = np.zeros(n)
    semis[voiced] = 12 * np.log2(pitches[voiced] / 440.0)

    vidx = np.flatnonzero(voiced)
    fixed = semis.copy()
    for pos, i in enumerate(vidx):
        med = np.median(semis[vidx[max(0, pos - octave_window):pos + octave_window + 1]])
        diff = semis[i] - med
        if abs(diff) > 7.0:
            fixed[i] = semis[i] - 12.0 * np.round(diff / 12.0)
    semis = fixed

    smoothed = semis.copy()
    for i in vidx:
        if 0 < i < n - 1 and voiced[i - 1] and voiced[i + 1]:
            smoothed[i] = np.median(semis[i - 1:i + 2])
    semis = smoothed

    i = 0
    while i < n:
        if voiced[i]:
            i += 1
            continue
        j = i
        while j < n and not voiced[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n and gap <= max_gap and abs(semis[j] - semis[i - 1]) < 2.0:
            semis[i:j] = np.linspace(semis[i - 1], semis[j], gap + 2)[1:-1]
            confidence[i:j] = 0.5 * min(confidence[i - 1], confidence[j])
            voiced[i:j] = True
        i = j

    out = np.zeros(n)
    out[voiced] = 440.0 * 2.0 ** (semis[voiced] / 12.0)
    return out, confidence


def detect_pitch_track(audio, config):
    """
    Full pitch-analysis stage used by run_pipeline.

    audio: 1-D detection signal (optionally pre-emphasised).
    Frames it with the SAME centres as the synthesis frames but WITHOUT the
    Hann window (YIN needs the raw waveform; windowing tilts d(tau)).

    Returns (pitches_hz, confidence), both shape (num_frames,); pitches is 0
    wherever confidence is 0.
    """
    from .framing import frame_signal
    frames, _ = frame_signal(audio, config, apply_window=False)
    raw_pitch, aperiodicity = detect_pitch_yin(frames, config.sample_rate,
                                               config.pitch_fmin, config.pitch_fmax)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    confidence = voicing_confidence(aperiodicity, frame_rms)
    return clean_pitch_track(raw_pitch, confidence)


def detect_pitch_for_all_frames(frames, sample_rate, fmin=80, fmax=1000):
    """
    Kept for backward compatibility (CONTRACTS.md). Now YIN-based: one pitch
    per frame in Hz, 0.0 where the frame isn't clearly periodic. Works on
    windowed frames too, though the pipeline uses detect_pitch_track
    (unwindowed frames + cleanup).
    """
    pitches, aperiodicity = detect_pitch_yin(frames, sample_rate, fmin, fmax)
    frame_rms = np.sqrt(np.mean(np.asarray(frames, dtype=np.float64) ** 2, axis=1))
    conf = voicing_confidence(aperiodicity, frame_rms)
    return np.where(conf > 0, pitches, 0.0)
