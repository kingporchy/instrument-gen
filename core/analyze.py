"""Fragmeant-style beat analyzer: Phase 2 of the Instrument Generator spec.

Pure numpy/scipy DSP (decode via ffmpeg CLI for non-wav). Emits: tempo (+grid
phase), downbeat, beat frames, pooled chroma layers (semitone-comb fold,
noise-floor subtraction, HPSS harmonic separation, tonal-frame weighting),
key (Krumhansl profiles with bass boost), chord timeline (Viterbi over
normalized templates; standalone bar-phase = fewest transitions), groove
(16th-slot bins, latency-calibrated microtiming, swing ratio).
Validated against synthetic truth tracks (tests/mktest.py: known tempo, key,
4-bar chord loop, backbeat with bar accents).
"""
import json, os, subprocess, sys, tempfile
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d, median_filter

SR = 22050
HOP = 256
NFFT = 1024
HOP_HI = 512
NFFT_HI = 4096
FPS = SR / HOP
FPS_HI = SR / HOP_HI
PPQ = 480
PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# ---------------------------------------------------------------- audio I/O
def load_audio(path):
    path = os.path.expanduser(path)
    if path.lower().endswith(".aiff"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR), tmp.name],
                       capture_output=True, check=True)
        path = tmp.name
    if path.lower().endswith(".wav"):
        sr, x = wavfile.read(path)
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            n = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, 1, n), np.arange(len(x)), x).astype(np.float32)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR), tmp.name],
                           capture_output=True, check=True)
            sr, x = wavfile.read(tmp.name)
            x = x.astype(np.float32)
            if x.ndim > 1:
                x = x.mean(axis=1)
        finally:
            os.unlink(tmp.name)
    x = x / max(1e-9, np.abs(x).max())
    return SR, x

# ---------------------------------------------------------------- spectra
def magnitude_spectrogram(x, nfft=NFFT, hop=HOP):
    frames = 1 + (len(x) - nfft) // hop
    win = np.hanning(nfft).astype(np.float32)
    out = np.empty((frames, nfft // 2 + 1), np.float32)
    for i in range(frames):
        out[i] = np.abs(rfft(x[i * hop: i * hop + nfft] * win))
    return out

def onset_flux(mag, smooth=31):
    lg = np.log1p(1000 * mag)
    d = np.maximum(0.0, lg[1:] - lg[:-1]).sum(axis=1)
    d = np.concatenate([[0.0], d])
    d = np.maximum(0.0, d - uniform_filter1d(d, smooth))
    return d / (d.max() + 1e-9)

# ---------------------------------------------------------------- tempo
def estimate_tempo(flux, host_bpm=None, lo=60.0, hi=200.0):
    span = np.arange(lo, hi, 0.5)
    lags = FPS * 60.0 / span
    f = flux - flux.mean()
    n = len(f)
    scores = np.zeros(len(span))
    norm = np.linalg.norm(f) + 1e-9
    for i, lag in enumerate(lags):
        li = int(round(lag))
        if li < 1 or li > n // 2:
            continue
        a, b = f[:-li], f[li:]
        scores[i] = np.dot(a, b) / (norm * (np.linalg.norm(b) + 1e-9))
    best = int(np.argmax(scores))
    bpm = span[best]
    def score_at(b):
        j = int(round((b - lo) / 0.5))
        return scores[j] if 0 <= j < len(scores) else 0.0
    if bpm > 150 and score_at(bpm / 2) > 0.85 * score_at(bpm):
        bpm /= 2
    if bpm < 70 and score_at(bpm * 2) > 0.85 * score_at(bpm):
        bpm *= 2
    return float(bpm), float(scores[best])

def _comb_grid_mean(flux, period, tol=3, phases=24):
    n = len(flux)
    best_v, best_phi = 0.0, 0.0
    for phi in np.linspace(0, period, phases, endpoint=False):
        idx = np.arange(phi, n, period).astype(int)
        if len(idx) < 4:
            continue
        lo = np.maximum(0, idx - tol)
        hi = np.minimum(n - 1, idx + tol)
        vals = np.array([flux[lo[j]: hi[j] + 1].max() for j in range(len(idx))])
        v = float(vals.mean())
        if v > best_v:
            best_v, best_phi = v, phi
    return best_phi, best_v

def tempo_from_flux(flux, flux_low, host_bpm=None):
    """Candidate ladder rescored on the LOW-BAND flux (sub-150 Hz: kicks
    plus bass roots). A true beat grid aligns with the low-frequency pulse;
    hat lattices and half-tempo kick patterns lose on this channel. With a
    host (DAW) tempo hint, prefer any credible nearby candidate."""
    bpm0, _ = estimate_tempo(flux, host_bpm)
    P0 = FPS * 60.0 / bpm0
    cands = {}
    for m in (1.0, 0.5, 2.0, 2 / 3, 1.5, 0.75, 4 / 3, 1 / 3, 3.0):
        P = P0 * m
        if not (8 <= P <= 400):
            continue
        for d in (0.97, 1.0, 1.03):
            Pd = P * d
            bpm = 60.0 * FPS / Pd
            if 55 <= bpm <= 190:
                cands[round(Pd, 4)] = bpm
    scored = sorted(((_comb_grid_mean(flux_low, Pd)[1], bpm) for Pd, bpm in cands.items()),
                    reverse=True)
    if not scored:
        return bpm0, 0.0
    s_best, bpm_best = scored[0]
    conf = s_best / (s_best + 0.3) if s_best > 0 else 0.0
    if host_bpm:
        near = [x for x in scored if abs(x[1] - host_bpm) <= host_bpm * 0.03]
        if near and near[0][0] >= 0.5 * s_best:
            # The host grid IS the truth: snap tempo exactly to the host value.
            bpm_best, conf = float(host_bpm), min(1.0, near[0][0] / (s_best + 1e-9) * 1.4)
    return float(bpm_best), float(conf)

def _peak_positions(f, rel_th=0.20):
    th = rel_th * max(1e-9, f.max())
    pk = ((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:]) & (f[1:-1] > th)).nonzero()[0] + 1
    return pk

def beat_grid(flux, bpm, use_low=None):
    period = FPS * 60.0 / bpm
    src = flux if use_low is None else use_low
    # Event-anchored phase: low-band onset peaks modulo the beat period cluster
    # at true beat positions; a sloppy windowed-max mean can lock BETWEEN events.
    pk = _peak_positions(src)
    if len(pk) >= 4:
        ph = (pk % period) / period * 24.0           # 24-bin histogram
        hist = np.bincount(ph.astype(int) % 24, minlength=24)
        k = int(np.argmax(hist))
        members = ((pk % period) >= k * period / 24 - period / 48) &                   ((pk % period) < (k + 1) * period / 24 + period / 48)
        phi = float((pk[members] % period).mean())
    else:
        phi, _ = _comb_grid_mean(src, period)
    n_beats = int(len(flux) // period)
    frames = [phi + k * period for k in range(n_beats) if phi + k * period < len(flux)]
    return phi, period, np.array(frames)

def band_accents(band_flux, phi, period):
    """Mean local-max (±2 frames) at each of the 4 bar phases -> (4,) accents."""
    out = []
    for j in range(4):
        idx = np.arange(phi + j * period, len(band_flux), period * 4).astype(int)
        idx = idx[idx < len(band_flux)]
        if len(idx) == 0:
            out.append(0.0)
            continue
        lo = np.maximum(0, idx - 2)
        hi = np.minimum(len(band_flux) - 1, idx + 2)
        vals = np.array([band_flux[lo[k]: hi[k] + 1].max() for k in range(len(idx))])
        out.append(float(vals.mean()))
    return np.array(out)

# ---------------------------------------------------------------- pitch layers
_COMB_CACHE = {}
def semitone_comb(nfft, lo_hz=90.0, hi_hz=2500.0, sigma_frac=0.028):
    """(n_bins, 12) matrix: each semitone position smears gaussianly over bins.
    Avoids nearest-bin snapping ghosts (±1 semitone leak ~11% instead of 100%)."""
    if nfft in _COMB_CACHE:
        return _COMB_CACHE[nfft]
    nb = nfft // 2 + 1
    freqs = np.arange(nb) * (SR / nfft)
    W = np.zeros((nb, 12), np.float32)
    k = 48          # midi note walk from C3-ish upward
    while True:
        f_k = 440.0 * 2 ** ((k - 69) / 12)
        if f_k > hi_hz:
            break
        if f_k >= lo_hz:
            sig = f_k * sigma_frac
            b_lo = int(max(0, np.floor((f_k - 3 * sig) / (SR / nfft))))
            b_hi = int(min(nb - 1, np.ceil((f_k + 3 * sig) / (SR / nfft))))
            for b in range(b_lo, b_hi + 1):
                W[b, k % 12] += np.exp(-0.5 * ((freqs[b] - f_k) / sig) ** 2)
        k += 1
    _COMB_CACHE[nfft] = W
    return W

def hpss_harmonic_ratio(clean):
    """Layout (time, freq): harmonic = median along TIME; percussive along FREQ."""
    H = median_filter(clean, size=(61, 1))
    P = median_filter(clean, size=(1, 31))
    mh = (H * H) / (H * H + P * P + 1e-9)
    return mh

def prepare_pitch_layers(mag_hi, floor_pct=5, nfft=NFFT_HI):
    """One shared pass: floor subtraction -> HPSS -> log -> semitone-comb fold.
    Returns (F_full, F_bass, tonal_frac): (frames,12) pitch-class energies for
    the full 90-2500 Hz band and the 90-250 Hz bass band, plus per-frame
    harmonic fraction (tonal-ness)."""
    floor = np.percentile(mag_hi, floor_pct, axis=0)
    clean = np.maximum(mag_hi - floor, 0.0)
    mh = hpss_harmonic_ratio(clean)
    lg = np.log1p(30.0 * clean * mh)
    W = semitone_comb(nfft)
    nb = mag_hi.shape[1]
    freqs = np.arange(nb) * (SR / nfft)
    full_band = (freqs >= 90) & (freqs <= 2500)
    bass_band = (freqs >= 90) & (freqs <= 250)
    F_full = (lg * full_band) @ W
    F_bass = (lg * bass_band) @ W
    tonal = mh.mean(axis=1)
    return F_full, F_bass, tonal

def pool_windows(F, tonal, ranges, tonal_thresh=0.30):
    """Equal-weight mean of per-frame normalized pitch distributions over
    (start,end) frame ranges, using only sufficiently-tonal frames; the
    global scale of each frame divides out loudness/decay bias."""
    nrm = np.linalg.norm(F, axis=1)
    alive = nrm > nrm.max() * 0.02
    tonal = tonal > tonal_thresh
    use = alive & tonal
    dist = F / (nrm[:, None] + 1e-9)
    out = np.zeros((len(ranges), 12), np.float32)
    for bi, (i0, i1) in enumerate(ranges):
        i0 = max(0, int(i0)); i1 = min(len(F), int(i1))
        if i1 <= i0:
            continue
        m = use[i0:i1]
        if m.sum() < 2:
            continue
        acc = dist[i0:i1][m].mean(axis=0)
        # re-normalize outliers (floor-sub residual noise frames have near-flat
        # dist; mostly they are excluded by `tonal`, keep a light guard)
        nz = np.linalg.norm(acc)
        if nz > 0:
            out[bi] = acc / nz
    return out

KEY_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KEY_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def key_from_chroma(key_full, key_bass):
    """Krumhansl profiles on pooled tonal-gated chroma (whole-clip pooled mean
    for both bands), bass band boosted 0.7 — roots anchor the tonic. Returns
    (key, confidence, ranked candidates) for chord-grammar re-ranking."""
    acc = np.maximum(key_full, 0.0); acc /= (acc.max() + 1e-9)
    acc_b = np.maximum(key_bass, 0.0); acc_b /= (acc_b.max() + 1e-9)
    acc = acc + 0.7 * acc_b
    acc /= (acc.max() + 1e-9)
    cands = []
    for tonic in range(12):
        r = np.roll(acc, -tonic)
        for prof, mode in ((KEY_MAJ, "major"), (KEY_MIN, "minor")):
            c = float(np.corrcoef(r, prof)[0, 1])
            cands.append((c, tonic, mode))
    cands.sort(key=lambda x: -x[0])
    c, tonic, mode = cands[0]
    return {"tonic": PITCHES[tonic], "mode": mode}, c, cands

def key_grammar_refine(cands, chords, weight=0.35):
    """Disambiguate profile ties (e.g. F# minor vs its relative/scales with a
    non-diatonic detected chord) via harmonic grammar: keys whose scale
    contains every detected chord's tones score higher. Chord tones are
    read from the chord-quality templates, not name guessing."""
    if not chords:
        c, tonic, mode = cands[0]
        return {"tonic": PITCHES[tonic], "mode": mode}, c
    chord_pcs = []
    for seg in chords:
        root = PITCHES.index(seg["root"])
        tones = {root} | {(root + i) % 12 for i, v in enumerate(CHORD_TEMPLATES[seg["quality"]]) if v > 0}
        chord_pcs.append(tones)
    scored = []
    for c, tonic, mode in cands:
        scale = set((tonic + s) % 12 for s in
                    ((0, 2, 4, 5, 7, 9, 11) if mode == "major"
                     else (0, 2, 3, 5, 7, 8, 10)))
        in_key = sum(1.0 if t <= scale else 0.3 for t in chord_pcs) / len(chord_pcs)
        scored.append((0.65 * c + weight * (in_key - 0.65), tonic, mode))
    scored.sort(key=lambda x: -x[0])
    s, tonic, mode = scored[0]
    return {"tonic": PITCHES[tonic], "mode": mode}, s

# ---------------------------------------------------------------- chords
def third_fifth(root, quality):
    t = {iv: v for iv, v in enumerate(CHORD_TEMPLATES[quality]) if v > 0}
    return t.get(3), t.get(4), t.get(7)

def diatonicize(chords, chroma_segments, key, step_ppq, escape=0.25):
    """Key-aware post-pass: penalize chords whose tones leave the key scale
    (relative-key ambiguity like Dm-vs-Dmaj7 inside F# minor resolves because
    F is NOT diatonic while C# (maj7) is). Chord tones from templates, not
    name guessing; escape weight keeps borrowed chords possible."""
    tonic = PITCHES.index(key["tonic"])
    steps = (0, 2, 4, 5, 7, 9, 11) if key["mode"] == "major" else (0, 2, 3, 5, 7, 8, 10)
    scale = set((tonic + s) % 12 for s in steps)
    states = [q for q in CHORD_TEMPLATES]
    out = []
    for seg, row in zip(chords, chroma_segments):
        pc = PITCHES.index(seg["root"])
        ivs = [i for i, v in enumerate(CHORD_TEMPLATES[seg["quality"]]) if v > 0]
        in_scale = sum(1 for iv in ivs if (pc + iv) % 12 in scale) / len(ivs)
        if in_scale >= 0.99:
            out.append(seg)
            continue
        best = (None, -1.0)
        for q in states:
            t = {i: v for i, v in enumerate(CHORD_TEMPLATES[q]) if v > 0}
            ivs_q = sorted(t)
            if sum(1 for iv in ivs_q if (pc + iv) % 12 in scale) / len(ivs_q) < 0.99:
                continue  # candidate must be diatonic at this same root
            v = np.array(CHORD_TEMPLATES[q], np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            c = float(np.dot(np.roll(v, pc), row)) * escape * in_scale
            if c > best[1]:
                best = ({**seg, "quality": q}, c)
        if best[0] is not None and best[1] > 0.1:
            out.append(best[0])
        else:
            out.append(seg)
    merged = []
    for s in out:
        if merged and s["root"] == merged[-1]["root"] and s["quality"] == merged[-1]["quality"]:
            merged[-1]["lengthPpq"] += s["lengthPpq"]
        else:
            merged.append(s)
    return merged

CHORD_TEMPLATES = {
    "maj":  [1.3, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min":  [1.3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "7":    [1.3, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.7, 0],
    "min7": [1.3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0.7, 0],
    "maj7": [1.3, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.7],
    "sus4": [1.3, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "dim":  [1.2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "aug":  [1.2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
}
CHORD_PRIOR = {"maj": 1.0, "min": 1.0, "7": 0.85, "min7": 0.85, "maj7": 0.85,
               "sus4": 0.8, "dim": 0.7, "aug": 0.5}

def chord_timeline(segments, step_ppq=PPQ, bass=None):
    """Viterbi-smoothed chord recognition over pooled chroma segments.
    bass: optional matrix (same shape) of bass-band chroma per segment — adds
    a root-evidence bonus to emissions, resolving subset confusions (e.g.
    F#m vs Dmaj7 share F#-A-C#; the sustained bass root disambiguates)."""
    N = len(segments)
    if N == 0:
        return []
    states = list(CHORD_TEMPLATES)
    Tn = {q: np.array(CHORD_TEMPLATES[q], np.float32) /
             (np.linalg.norm(CHORD_TEMPLATES[q]) + 1e-9) for q in states}
    emis = np.zeros((N, len(states)), np.float32)
    roots = np.zeros((N, len(states)), int)
    for i in range(N):
        bvec = bass[i] if bass is not None else None
        bmax = float(np.max(bvec)) if (bass is not None and bvec is not None) else 0.0
        for si, q in enumerate(states):
            best, best_r = 0.0, 0
            for r in range(12):
                c = float(np.dot(np.roll(Tn[q], r), segments[i]))
                if bmax > 0.05:
                    c += 0.30 * float(bvec[r]) / bmax
                if c > best:
                    best, best_r = c, r
            emis[i, si] = max(best * CHORD_PRIOR[q], 0.0)
            roots[i, si] = best_r
    STAY_BONUS, MOVE_PENALTY = 0.02, 0.05
    dp = emis[0].copy()
    back = np.zeros((N, len(states)), int)
    for i in range(1, N):
        prev = dp.copy()
        dpn = np.empty(len(states), np.float32)
        for s in range(len(states)):
            stay = prev[s] + STAY_BONUS + emis[i, s]
            order = np.argsort(prev)[::-1]
            mo = int(order[0])
            if mo == s and len(order) > 1:
                mo = int(order[1])
            mv = prev[mo] - MOVE_PENALTY + emis[i, s]
            if stay >= mv:
                dpn[s] = stay; back[i, s] = s
            else:
                dpn[s] = mv; back[i, s] = mo
        dp = dpn
    path = [int(np.argmax(dp))]
    for i in range(N - 1, 0, -1):
        path.append(back[i, path[-1]])
    path.reverse()
    merged = []
    for i, si in enumerate(path):
        q = states[si]
        seg = {"startPpq": i * step_ppq, "lengthPpq": step_ppq,
               "root": PITCHES[roots[i, si]], "quality": q}
        if merged and seg["root"] == merged[-1]["root"] and seg["quality"] == merged[-1]["quality"]:
            merged[-1]["lengthPpq"] += step_ppq
        else:
            merged.append(seg)
    return merged

# ---------------------------------------------------------------- groove
def groove_and_swing(flux, beat_frames_, bpm, downbeat=0):
    period = FPS * 60.0 / bpm
    sixteenth = period / 4.0
    n = len(flux)
    th = 0.28 * max(1e-9, flux.max())
    pk = ((flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > th)).nonzero()[0] + 1
    if len(pk) == 0:
        return {"swingRatio": 1.0, "bins": [0.0] * 16, "microMs": [0.0] * 16}
    bar0 = beat_frames_[downbeat] if downbeat < len(beat_frames_) else beat_frames_[0]
    slot = (pk - bar0) / sixteenth
    slot_i = np.round(slot).astype(int) % 16
    bins = np.zeros(16)
    for s, w in zip(slot_i, flux[pk]):
        bins[s] += w
    bins /= (bins.max() + 1e-9)

    def pk_pos(i):
        a, b = flux[i - 1], flux[i]
        c = flux[i + 1] if i + 1 < n else flux[i]
        d = (a - c) / (2 * (a - 2 * b + c) + 1e-12)
        return i + float(np.clip(d, -1, 1))

    pos = np.array([pk_pos(int(i)) for i in pk])
    offs_raw = ((pos - bar0) - np.round((pos - bar0) / sixteenth) * sixteenth) / FPS * 1000.0
    latency_ms = float(np.median(offs_raw))
    pos = pos - latency_ms / 1000.0 * FPS
    offs_ms = ((pos - bar0) - np.round((pos - bar0) / sixteenth) * sixteenth) / FPS * 1000.0
    micro = np.zeros(16); cnt = np.zeros(16)
    for s, o in zip(slot_i, offs_ms):
        micro[s] += o; cnt[s] += 1
    micro = [float(micro[s] / cnt[s]) if cnt[s] else 0.0 for s in range(16)]

    ratios = []
    for m in range(1, len(beat_frames_) - 1):
        if (m - downbeat) % 4 not in (0, 1):
            continue
        t_on = beat_frames_[m]
        t_off_exp = t_on + period / 2.0
        mask = np.abs(pk - t_off_exp) < period / 8.0
        if not mask.any():
            continue
        near_idx = np.nonzero(mask)[0]
        j = int(near_idx[np.argmax(flux[pk[mask]])])
        tpk = pos[j]
        ratios.append((tpk - t_on) / (t_on + period - tpk))
    swing_ratio = float(np.median(ratios)) if ratios else 1.0
    return {"swingRatio": round(swing_ratio, 3),
            "bins": [round(float(x), 3) for x in bins],
            "microMs": [round(float(x), 1) for x in micro]}

# ---------------------------------------------------------------- driver
def analyze_signal(x, host_bpm=None, bar_beat_index=None):
    """Full chain. bar_beat_index: 0..3 clip-start position in a bar when the
    host provides it; None -> self-supervised: choose the bar phase whose
    chord timeline has the fewest transitions (ties: loudest low-band accent).
    """
    mag = magnitude_spectrogram(x, NFFT, HOP)
    flux = onset_flux(mag)
    flux_low = onset_flux(mag[:, :max(2, int(110.0 / (SR / NFFT)))], smooth=7)
    bpm, conf = tempo_from_flux(flux, flux_low, host_bpm)
    phi, period, bframes = beat_grid(flux, bpm, use_low=flux_low)
    nb = len(bframes) // 4
    mag_hi = magnitude_spectrogram(x, NFFT_HI, HOP_HI)
    F_full, F_bass, tonal = prepare_pitch_layers(mag_hi)
    key_full = pool_windows(F_full, tonal, [(0, len(F_full))])[0]
    key_bass = pool_windows(F_bass, tonal, [(0, len(F_bass))])[0]
    key, kconf, key_cands = key_from_chroma(key_full, key_bass)
    hi_scale = FPS_HI / FPS   # coarse frames -> hi frames (hi grid is FINER: multiply by 0.5)
    bar_len_hi = 4 * period * hi_scale

    def bar_windows(j):
        out = []
        i = int(j)
        while i + 4 <= len(bframes):
            s = bframes[i] * hi_scale
            out.append((int(s), int(s + bar_len_hi)))
            i += 4
        return out

    db = int(bar_beat_index) % 4 if bar_beat_index is not None else None
    if nb >= 2:
        if db is None:
            # Accent-primary on the low band (kick/bass onsets). Chord-set
            # phase scoring is unreliable standalone (nested chords straddle
            # better than truth); the plugin supplies bar_beat_index instead.
            acc_low = band_accents(flux_low, phi, period)
            db = int(np.argmax(acc_low))
        windows = bar_windows(db)
        bar_ch = pool_windows(F_full, tonal, windows)
        bar_bass = pool_windows(F_bass, tonal, windows)
        chords = chord_timeline(bar_ch, step_ppq=4 * PPQ, bass=bar_bass)
        chords = diatonicize(chords, bar_ch, key, 4 * PPQ)
        bframes = bframes[db:]
    else:
        ranges = [(int(bt * hi_scale) - 3, int(bt * hi_scale) + 4) for bt in bframes]
        bch = pool_windows(F_full, tonal, ranges)
        bbass = pool_windows(F_bass, tonal, ranges)
        chords = chord_timeline(bch, step_ppq=PPQ, bass=bbass)
        chords = diatonicize(chords, bch, key, PPQ)
        if db is None:
            db = int(np.argmax(band_accents(flux_low, phi, period)))
        bframes = bframes[db:] if db < len(bframes) else bframes

    if key_cands is not None:
        key, kconf = key_grammar_refine(key_cands, chords)
    groove = groove_and_swing(flux, bframes, bpm, 0)
    bars = max(1, len(bframes) // 4)
    return {
        "tempo": round(bpm, 1), "tempoConfidence": round(conf, 3),
        "timeSignature": [4, 4], "downbeatIndex": db,
        "key": key, "keyConfidence": round(kconf, 3),
        "chords": chords, "bars": bars,
        "groove": groove,
        "analysisParams": {"sr": SR, "hop": HOP, "nfftHi": NFFT_HI, "ppq": PPQ},
    }

def analyze(path, host_bpm=None, t_from=None, t_to=None, bar_beat_index=None):
    sr, x = load_audio(path)
    if t_from is not None:
        x = x[int(t_from * SR): int((t_to if t_to else 1e12) * SR)]
    result = analyze_signal(x, host_bpm, bar_beat_index=bar_beat_index)
    result["source"] = os.path.basename(path)
    return result

if __name__ == "__main__":
    path = sys.argv[1]
    host_bpm = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(analyze(path, host_bpm), indent=2))
