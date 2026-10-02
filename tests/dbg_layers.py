"""Inter-layer debug dump for the analyzer on the truth track.
Prints per-stage 12-vectors at TRUE bar windows (known by construction) so we
can see exactly which stage destroys the pitch information.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/kingporchy/Documents/ChatGPT/New project/instrument-gen/core")
import analyze as A

P = A.PITCHES
sr, x = A.load_audio("/tmp/truth102.wav")

# Coarse grid: find true tempo/phase from construction (we KNOW the render used
# plain bar grids starting at t=0), but verify with detected bpm (host hint).
bpm = 102.0
period_c = A.FPS * 60.0 / bpm          # coarse frames per beat
period_h = A.FPS_HI * 60.0 / bpm       # hi frames per beat

mag_hi = A.magnitude_spectrogram(x, A.NFFT_HI, A.HOP_HI)

# Stage 1: raw fold (no floor, no HPSS)
pcs_v, freqs_v = A.pc_map(mag_hi.shape[1], A.NFFT_HI)
valid_v = (freqs_v >= 180) & (freqs_v <= 2200) & (pcs_v >= 0)

def fold(M, ranges, w=None):
    out = np.zeros((len(ranges), 12), np.float32)
    for bi, (i0, i1) in enumerate(ranges):
        i0 = max(0, i0); i1 = min(len(M), i1)
        if i1 <= i0: continue
        seg = M[i0:i1]
        acc = np.array([seg[:, pcs_v == p].sum() for p in range(12)])
        if w is not None:
            acc = acc  # weighting already applied in caller
        n = np.linalg.norm(acc)
        out[bi] = acc / n if n > 0 else 0
    return out

centers = []
for bar in range(4):
    beat_start_h = bar * 4 * period_h
    i0 = int(beat_start_h + 2.2 * period_h)   # skip attack+lag: pool middle 1.5 beats
    i1 = int(beat_start_h + 3.7 * period_h)
    centers.append((i0, i1))

floor = np.percentile(mag_hi, 40, axis=0)
clean = np.maximum(mag_hi - floor, 0.0)
harm = A.hpss_harmonic_mask(clean)
lg_raw = np.log1p(30.0 * mag_hi)
lg_clean = np.log1p(30.0 * clean)
lg_harm = np.log1p(30.0 * harm)
for name, M in (("raw", lg_raw), ("floored", lg_clean), ("hpss", lg_harm)):
    v = fold(M, centers)
    print(f"== {name} ==")
    for bi in range(4):
        top = np.argsort(v[bi])[::-1][:4]
        print(f"  bar{bi}:", [(P[t], round(float(v[bi, t]), 2)) for t in top])
print("TRUTH: bar0 F#m(F#,A,C#) bar1 Dmaj7(D,F#,A,C#) bar2 Bm(B,D,F#) bar3 C#m(C#,E,G#)")

# probe single bin: B3 (midi 59, ~494 Hz) over time — should peak only in bar 2
b_bin = int(round(494.0 / (A.SR / A.NFFT_HI)))
ts = lg_harm[:, b_bin]
print(f"\nB3 bin {b_bin} harmonic-log per-bar mean (truth: bar2 only):")
for bar in range(4):
    i0 = int(bar * 4 * period_h); i1 = int((bar + 1) * 4 * period_h)
    print(f"  bar{bar}: {ts[i0:i1].mean():.3f}")
