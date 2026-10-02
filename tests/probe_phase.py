"""Probe WHY phase selection/key/tempo mis-grade despite good bar chroma."""
import sys
import numpy as np

sys.path.insert(0, "/Users/kingporchy/Documents/ChatGPT/New project/instrument-gen/core")
import analyze as A

P = A.PITCHES
sr, x = A.load_audio("/tmp/truth102.wav")

mag = A.magnitude_spectrogram(x, A.NFFT, A.HOP)
flux = A.onset_flux(mag)
bpm, conf = A.tempo_from_flux(flux, 102.0)
phi, period, bframes = A.beat_grid(flux, bpm)
mag_hi = A.magnitude_spectrogram(x, A.NFFT_HI, A.HOP_HI)
F_full, F_bass, tonal = A.prepare_pitch_layers(mag_hi)
hi_scale = A.FPS / A.FPS_HI
period_hi = period * hi_scale

print("tempo(with hint):", bpm, "| beat period coarse:", round(period, 1), "frames")
print("tonal>0.30 fraction:", round(float((tonal > 0.30).mean()), 3))

# KEY vectors as the pipeline actually computes them
kf = A.pool_windows(F_full, tonal, [(0, len(F_full))])[0]
kb = A.pool_windows(F_bass, tonal, [(0, len(F_bass))])[0]
acc = np.maximum(kf, 0); acc /= (acc.max() + 1e-9)
acb = np.maximum(kb, 0); acb /= (acb.max() + 1e-9)
comb = acc + 0.7 * acb; comb /= comb.max()
print("\nkey_full (C..B):", dict(zip(P, np.round(acc, 2))))
print("key_bass (C..B):", dict(zip(P, np.round(acb, 2))))
print("combined       :", dict(zip(P, np.round(comb, 2))))
# explicit profile comparison: F#-minor vs B-minor correlation on `comb`
r_fsharp = np.roll(comb, -P.index("F#"))
r_b = np.roll(comb, -P.index("B"))
print("corr F# minor:", round(float(np.corrcoef(r_fsharp, A.KEY_MIN)[0, 1]), 3),
      "| corr B minor:", round(float(np.corrcoef(r_b, A.KEY_MIN)[0, 1]), 3))

# PHASE scoring exactly as the driver does
Tn = {q: np.array(A.CHORD_TEMPLATES[q], np.float32) /
         (np.linalg.norm(A.CHORD_TEMPLATES[q]) + 1e-9) for q in A.CHORD_TEMPLATES}
def bar_windows(j):
    out = []
    i = int(j)
    while i + 4 <= len(bframes):
        s = bframes[i] * hi_scale
        out.append((int(s), int(s + 4 * period_hi)))
        i += 4
    return out

print("\nPHASE SCORING (truth: j=0):")
for j in range(4):
    segs = A.pool_windows(F_full, tonal, bar_windows(j))
    fits = []
    for row in segs:
        if row.sum() == 0:
            continue
        fits.append(max(float(np.dot(np.roll(Tn[q], r), row))
                        for q in A.CHORD_TEMPLATES for r in range(12)))
    tl = A.chord_timeline(segs, step_ppq=4 * A.PPQ)
    mean_fit = float(np.mean(fits)) if fits else 0.0
    print(f"  j={j}: windows={len(segs)} mean_fit={mean_fit:.3f} "
          f"transitions={len(tl)} score={mean_fit - 0.02*max(0,len(tl)-1):.3f}")

# TEMPO support criterion candidates
print("\nTEMPO candidates (support = frac of grid pts with windowed max > 0.15 max):")
fmax = flux.max()
for bpm_c in (68.0, 90.7, 102.0, 136.0, 153.0):
    Pd = A.FPS * 60.0 / bpm_c
    sup = []
    mx = []
    for ph in np.linspace(0, Pd, 24, endpoint=False):
        idx = np.arange(ph, len(flux), Pd).astype(int)
        if len(idx) < 4: continue
        lo = np.maximum(0, idx - 3); hi = np.minimum(len(flux) - 1, idx + 3)
        vals = np.array([flux[lo[k]: hi[k] + 1].max() for k in range(len(idx))])
        sup.append(float((vals > 0.15 * fmax).mean()))
        mx.append(float(vals.mean()))
    print(f"  {bpm_c:6.1f} BPM  best-phase support={max(sup):.2f} meanmax={max(mx):.3f}")
