"""Render a synthetic test track with KNOWN truth, to grade the analyzer.
Truth: 102.0 BPM, key F# minor, chord loop F#m - Dmaj7 - Bm - C#m
(1 bar each, harmonic-stacked tones + bass roots), straight (no swing),
kick on 1/3 with bar-1 accent, snare on 2/4, hats on off-8ths.
"""
import numpy as np
from scipy.io import wavfile

SR = 22050
BPM = 102.0
BEAT = 60.0 / BPM
BARS = 8
DUR = BARS * 4 * BEAT + 1.0
N = int(DUR * SR)
t_all = np.arange(N) / SR
audio = np.zeros(N, np.float32)

def add_kick(t0, f=52.0, dur=0.28, amp=0.9):
    i0 = int(t0 * SR); n = int(dur * SR)
    if i0 + n > N: n = N - i0
    t = np.arange(n) / SR
    env = np.exp(-9 * t / dur)
    sweep = f * np.exp(-30 * t) + 34
    audio[i0:i0+n] += (np.sin(2*np.pi*np.cumsum(sweep)/SR) * env * amp).astype(np.float32)

def add_snare(t0, amp=0.5):
    """Real snares are band-passed: high-pass the noise (1-pole via diff)."""
    i0 = int(t0 * SR); n = int(0.18 * SR)
    if i0 + n > N: n = N - i0
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(noise, prepend=noise[0]).astype(np.float32)  # HP: kill sub content
    env = np.exp(-16 * np.arange(n) / (0.18*SR))
    audio[i0:i0+n] += (noise * env * amp).astype(np.float32)

def add_hat(t0, amp=0.25):
    i0 = int(t0 * SR); n = int(0.05 * SR)
    if i0 + n > N: n = N - i0
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(np.diff(noise, prepend=noise[0]), prepend=noise[0]).astype(np.float32)  # double HP
    env = np.exp(-60 * np.arange(n) / (0.05*SR))
    audio[i0:i0+n] += (noise * env * amp).astype(np.float32)

NOTE = {p: i for i, p in enumerate(["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"])}
def pc(p): return NOTE[p] % 12

def add_chord(t0, dur, root, quality, amp=0.30):
    tones = {"min": [0,3,7], "7": [0,4,7,10], "maj7": [0,4,7,11]}[quality]
    r = pc(root) + 60  # structure around C4
    rb = pc(root) + 48 # BASS ROOT around C3 — how real music anchors chords
    i0 = int(t0 * SR); n = int(dur * SR)
    if i0 + n > N: n = N - i0
    t = np.arange(n) / SR
    # bass root: sustained, slightly louder, slow attack
    f_b = 440.0 * 2 ** ((rb - 69) / 12)
    env_b = np.minimum(1.0, 12*t) * np.exp(-0.8 * t / dur)
    wave_b = np.zeros_like(t)
    for nh in range(1, 5):
        if f_b * nh < SR * 0.45:
            wave_b += np.sin(2*np.pi*f_b*nh*t) / nh
    audio[i0:i0+n] += (wave_b * env_b * amp * 0.9).astype(np.float32)
    for iv in tones:
        f = 440.0 * 2 ** ((r + iv - 69) / 12)
        env = np.minimum(1.0, 8*t) * np.exp(-2.0 * t / dur)
        # harmonic-stacked tone (sawtooth-lite): real instruments carry a
        # harmonic series; pure sines carry zero harmonic evidence and are a
        # pathological case for chroma recognition
        wave = np.zeros_like(t)
        for nh in range(1, 7):
            if f * nh < SR * 0.45:
                wave += np.sin(2*np.pi*f*nh*t) / nh
        audio[i0:i0+n] += (wave * env * amp / len(tones)).astype(np.float32)

# chord loop (diatonic to F# minor): F#m (1 bar) Dmaj7 (1) Bm (1) C#m (1)
prog = [("F#", "min"), ("D", "maj7"), ("B", "min"), ("C#", "min")]

def render():
    for bar in range(BARS):
        c = prog[bar % 4]
        t0 = bar * 4 * BEAT
        add_chord(t0, 4 * BEAT * 0.95, c[0], c[1])
        for b in range(4):
            if b in (0, 2):
                add_kick(t0 + b * BEAT, amp=1.0 if b == 0 else 0.8)  # bar-1 accent
            if b in (1, 3):
                add_snare(t0 + b * BEAT)
            add_hat(t0 + b * BEAT + BEAT / 2)
    peak = np.abs(audio).max()
    out = audio / peak * 0.9
    wavfile.write("/tmp/truth102.wav", SR, (out * 32767).astype(np.int16))
    print("rendered /tmp/truth102.wav", round(DUR, 1), "s")

if __name__ == "__main__":
    render()
