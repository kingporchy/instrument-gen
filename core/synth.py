"""Note-level synthesis for generated takes — no external samples needed.
Karplus-Strong plucked strings (guitar/bass), FM electric piano, detuned pad.
Velocity controls level + brightness; decay envelope + Schroeder-style reverb
+ soft limiter turn the diagnostic render into something listenable.

Not sample-perfect (that's the next phase), but each family has plausible
timbre, real string-decay physics (KS), and musical dynamics.
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


def _env_fade(out, sr, ms=8):
    n = int(sr * ms / 1000)
    if n > 0 and len(out) > 2 * n:
        out[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        out[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return out


def ks_pluck(freq, dur, sr, amp=1.0, bright=0.6, damping=0.996, seed=0):
    """Karplus-Strong plucked string, vectorized per period."""
    n_buf = max(2, int(round(sr / freq)))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1, 1, n_buf).astype(np.float32)
    a = 0.15 + 0.8 * bright          # one-pole lowpass on the excitation
    for i in range(1, n_buf):
        buf[i] = a * buf[i] + (1 - a) * buf[i - 1]
    total = int(dur * sr)
    periods = int(np.ceil(total / n_buf)) + 1
    out = np.empty(periods * n_buf, np.float32)
    for p in range(periods):
        out[p * n_buf:(p + 1) * n_buf] = buf
        buf = (damping * 0.5) * (buf + np.roll(buf, -1))
    out = out[:total]
    return _env_fade(out, sr) * amp


def ks_bass(freq, dur, sr, amp=1.0, bright=0.4, seed=0):
    return ks_pluck(freq, dur, sr, amp=amp, bright=bright, damping=0.9985, seed=seed)


def fm_ep(freq, dur, sr, amp=1.0, bright=0.5, seed=0):
    """Two-operator FM electric piano: bell-ish attack, mellow sustain."""
    t = np.arange(int(dur * sr)) / sr
    idx = (1.2 + 2.6 * bright) * np.exp(-3.5 * t) + 0.12
    out = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * 2 * freq * t))
    env = np.minimum(1.0, t * 150) * np.exp(-1.6 * t)
    return _env_fade((out * env * amp).astype(np.float32), sr)


def pad_tone(freq, dur, sr, amp=1.0, bright=0.5, seed=0):
    """Detuned additive pad: three slightly detuned partials + harmonics."""
    t = np.arange(int(dur * sr)) / sr
    out = np.zeros_like(t, dtype=np.float32)
    for det, k in ((0.9987, 1.0), (1.0, 1.0), (1.0013, 1.0), (2.0, 0.35), (3.0, 0.12)):
        out += np.sin(2 * np.pi * freq * k * det * t).astype(np.float32) / k
    env = np.minimum(1.0, t * 4) * np.exp(-0.35 * t)
    return _env_fade((out * env * amp * 0.5).astype(np.float32), sr)


SYNTHS = {
    "guitar": ks_pluck, "funk": ks_pluck, "house": ks_pluck,
    "bass": ks_bass, "piano": fm_ep, "pad": pad_tone,
}


def render_note(style, freq, dur, vel, sr, seed=0):
    """Velocity -> level and brightness; family dispatch by style name."""
    amp = 0.25 + 0.75 * (vel / 127.0)
    bright = 0.3 + 0.55 * (vel / 127.0)
    fn = SYNTHS.get(style, ks_pluck)
    return fn(freq, min(dur, 6.0), sr, amp=amp, bright=bright, seed=seed)


def reverb_wet(dry, sr, seconds=1.1, decay=3.2, wet=0.16, seed=1):
    """Exponentially-decaying-noise IR convolution: cheap, musical space."""
    n = int(sr * seconds)
    rng = np.random.default_rng(seed)
    ir = (rng.standard_normal(n) * np.exp(-decay * np.arange(n) / sr)).astype(np.float32)
    pre = int(0.012 * sr)
    ir[:pre] *= np.linspace(0, 1, pre, dtype=np.float32)
    ir /= np.sqrt((ir ** 2).sum() + 1e-9)
    return (fftconvolve(dry, ir)[: len(dry)] * wet).astype(np.float32)


def finalize(audio, sr):
    """Reverb + normalize + soft-clip limiter, returns int16 for wavfile."""
    audio = audio + reverb_wet(audio, sr)
    peak = max(1e-9, np.abs(audio).max())
    audio = audio / peak
    audio = np.tanh(audio * 1.6) / np.tanh(1.6)
    return (audio * 0.92 * 32767).astype(np.int16)
