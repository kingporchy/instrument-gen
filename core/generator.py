"""Instrument Generator — Phase 3 MIDI path (spec §5).
Generates a note list from prompt + analysis (key, chords, groove), applies
groove quantization, chord/key constraints, humanize; exports MIDI (mido)
and renders audio via a built-in sample-synth (generator.py synth section).
"""
import json, math, os, random
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

PPQ = 480

# --- prompt interpretation -------------------------------------------------
STYLE_PRESETS = {
    "funk":   {"range": (36, 84), "density": 0.55, "vel_range": (72, 108), "octave_bias": 48},
    "house":  {"range": (36, 96), "density": 0.45, "vel_range": (80, 112), "octave_bias": 48},
    "piano":  {"range": (36, 96), "density": 0.40, "vel_range": (60, 100), "octave_bias": 60},
    "guitar": {"range": (40, 88), "density": 0.50, "vel_range": (70, 105), "octave_bias": 52},
    "pad":    {"range": (36, 96), "density": 0.15, "vel_range": (50, 80),  "octave_bias": 60},
    "bass":   {"range": (28, 55), "density": 0.60, "vel_range": (80, 115), "octave_bias": 36},
}

def pick_preset(prompt):
    p = prompt.lower()
    for name, preset in STYLE_PRESETS.items():
        if name in p:
            return name, preset
    return "funk", STYLE_PRESETS["funk"]

def key_mask(key):
    major = [0, 2, 4, 5, 7, 9, 11]
    minor = [0, 2, 3, 5, 7, 8, 10]
    tonic = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"].index(key["tonic"])
    scale = major if key["mode"] == "major" else minor
    return tonic, set((tonic + s) % 12 for s in scale)

CHORD_TONES = {
    "maj":   [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10], "min7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11], "sus4": [0, 5, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
}
CHORD_EXTENSIONS = {"maj": [9], "min": [9], "7": [9], "min7": [9], "maj7": [9]}

def note_candidates(root_pc, quality, scale_set, lo, hi):
    tone_set = set()
    for iv in CHORD_TONES.get(quality, [0, 4, 7]):
        tone_set.add((root_pc + iv) % 12)
    for iv in CHORD_EXTENSIONS.get(quality, []):
        tone_set.add((root_pc + iv) % 12)
    in_scale = set(p for p in tone_set if p in scale_set)
    final = in_scale if in_scale else tone_set  # avoid out-of-key chord tones
    notes = []
    for pc in final:
        for octv in range(lo // 12, hi // 12 + 1):
            n = octv * 12 + pc
            if lo <= n <= hi:
                notes.append(n)
    return sorted(notes)

# --- groove quantization ---------------------------------------------------
def groove_quantize(start_beats, analysis, humanize):
    """Snap sixteenth-fraction positions toward groove bins (0..15)."""
    g = analysis["groove"]
    bins = np.array(g["bins"], dtype=np.float32)
    micro = np.array(g["microMs"], dtype=np.float32) if "microMs" in g else np.zeros(16)
    # weight: pull toward heavy bins, push away from empty ones
    snapped = []
    for s in start_beats:
        slot16 = (s * 4) % 16
        if bins[int(slot16)] > 0.35:
            pulled = s + (micro[int(slot16)] / 1000.0) * 0.35 * humanize
        else:
            pulled = s
        snapped.append(pulled)
    return snapped

# --- generation ------------------------------------------------------------
def generate(analysis, prompt, bars=4, seed=7, humanize=0.4):
    rnd = random.Random(seed)
    name, preset = pick_preset(prompt)
    lo, hi = preset["range"]
    tonic, scale = key_mask(analysis["key"])
    chords = analysis["chords"]
    total_beats = bars * 4
    # take chord timeline (ppq), limit to requested bars
    usable = [c for c in chords if c["startPpq"] < total_beats * PPQ]
    if not usable:
        usable = [{"startPpq": 0, "lengthPpq": total_beats * PPQ, "root": analysis["key"]["tonic"], "quality": "min"}]
    notes = []
    for c in usable:
        start_beat = c["startPpq"] / PPQ
        end_beat = (c["startPpq"] + c["lengthPpq"]) / PPQ
        seg_beats = min(end_beat, total_beats) - start_beat
        if seg_beats <= 0: continue
        root_pc = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"].index(c["root"])
        cand = note_candidates(root_pc, c["quality"], scale, lo, hi)
        if not cand: continue
        # rhythmic slots: on-beat, off-beat 8ths, 16ths per density
        density = preset["density"]
        density = preset["density"]
        offbeat = "off-beat" in prompt.lower() or "offbeat" in prompt.lower()
        n_slots = max(1, int(seg_beats * density * 2))  # ~two slots per beat at density 1
        slots = []
        for k in range(n_slots):
            if rnd.random() > density * 1.2 and k > 0: continue
            if offbeat:
                slot = start_beat + k + 0.5  # land off the beat
                if slot >= end_beat: continue
            else:
                slot = start_beat + k / max(1, n_slots) * seg_beats
            slots.append(slot)
        for s in slots:
            pitch = rnd.choice(cand)
            stabs = bool(preset["density"] > 0.4 and rnd.random() < 0.35)
            dur_beats = max(0.08, rnd.uniform(0.10, 0.30)) if stabs else rnd.uniform(0.4, max(0.5, seg_beats * 0.4))
            vel = rnd.randint(*preset["vel_range"])
            notes.append({"pitch": pitch, "startPpq": int(round(s * PPQ)),
                          "lengthPpq": int(round(dur_beats * PPQ)), "velocity": vel})
    # groove quantize + humanize
    beats = [n["startPpq"] / PPQ for n in notes]
    snapped = groove_quantize(beats, analysis, humanize)
    for n, sb in zip(notes, snapped):
        base = sb * PPQ
        jitter = rnd.uniform(-0.01, 0.01) * PPQ * humanize
        n["startPpq"] = max(0, int(round(base + jitter)))
        n["velocity"] = int(min(127, max(1, n["velocity"] + rnd.uniform(-8, 8) * humanize)))
    notes.sort(key=lambda n: (n["startPpq"], n["pitch"]))
    return name, notes

# --- MIDI export ------------------------------------------------------------
def export_midi(notes, path, tempo_bpm):
    mid = MidiFile(ticks_per_beat=PPQ)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(MetaMessage("track_name", name="Fragmeant take", time=0))
    events = []
    for n in notes:
        events.append((n["startPpq"], 1, n))
        events.append((n["startPpq"] + n["lengthPpq"], 0, n))
    events.sort(key=lambda e: (e[0], e[1]))
    t = 0
    for ticks, onoff, n in events:
        dt = max(0, ticks - t); t = ticks
        if onoff:
            track.append(Message("note_on", note=n["pitch"], velocity=n["velocity"], time=dt))
        else:
            track.append(Message("note_off", note=n["pitch"], velocity=0, time=dt))
    track.append(MetaMessage("end_of_track", time=0))
    mid.save(path)
    return path

# --- simple synth render ----------------------------------------------------
def render_via_fluidsynth(midi_path, wav_path, sf2_path, sr=44100, gain=0.6):
    """Render MIDI through FluidSynth + a real GM soundfont, then apply the
    shared reverb + limiter chain. Returns (path, seconds) like render_wav.
    Auto-detects the fluidsynth binary (FLUIDSYNTH_PATH env, or the project
    thirdparty prefix) and sets DYLD paths for its local framework.
    Raises RuntimeError if unavailable or rendering fails."""
    import subprocess, shutil, os
    from scipy.io import wavfile
    import synth as S
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("FLUIDSYNTH_PATH"),
        os.path.join(here, "..", "thirdparty", "fluidsynth", "bin", "fluidsynth"),
        "fluidsynth",
    ]
    fluid = None
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(c) and os.access(c, os.X_OK):
            fluid = os.path.abspath(c)
            break
        if "/" not in c and shutil.which(c):
            fluid = shutil.which(c)
            break
    if fluid is None:
        raise RuntimeError("fluidsynth binary not found")
    env = dict(os.environ)
    fw = os.path.join(os.path.dirname(os.path.dirname(fluid)), "Library", "Frameworks")
    if os.path.isdir(fw):
        env["DYLD_FRAMEWORK_PATH"] = fw
    raw = wav_path + ".raw.wav"
    cmd = [fluid, "-ni", "-F", raw, "-r", str(sr), "-g", str(gain), sf2_path, midi_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0 or not os.path.exists(raw):
        raise RuntimeError(f"fluidsynth failed: {proc.stderr[-400:]}")
    sr_actual, audio = wavfile.read(raw)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio / max(1e-9, np.abs(audio).max())
    out = S.finalize(audio, sr_actual)
    seconds = len(out) / sr_actual
    wavfile.write(wav_path, sr_actual, out)
    os.unlink(raw)
    return wav_path, seconds

def render_wav(notes, path, tempo_bpm, sr=22050, prompt="", style=None):
    """Render notes through the synth layer (synth.py): per-family synthesis,
    velocity-mapped brightness, convolution reverb, soft-clip limiter."""
    import synth as S
    style = style or pick_preset(prompt)[0]
    total_beats = max(n["startPpq"] + n["lengthPpq"] for n in notes) / PPQ + 1.5
    seconds = total_beats * 60.0 / tempo_bpm
    audio = np.zeros(int(seconds * sr), np.float32)
    for i, n in enumerate(notes):
        t0 = n["startPpq"] / PPQ * 60.0 / tempo_bpm
        dur = n["lengthPpq"] / PPQ * 60.0 / tempo_bpm
        f = 440.0 * (2 ** ((n["pitch"] - 69) / 12))
        tone = S.render_note(style, f, dur + 0.35, n["velocity"], sr, seed=i * 7 + 1)
        i0 = int(t0 * sr)
        i1 = min(len(audio), i0 + len(tone))
        if i1 <= i0:
            continue
        audio[i0:i1] += tone[: i1 - i0]
    from scipy.io import wavfile
    wavfile.write(path, sr, S.finalize(audio, sr))
    return path, seconds

if __name__ == "__main__":
    import sys
    analysis = json.load(open(sys.argv[1]))
    prompt = sys.argv[2] if len(sys.argv) > 2 else "warm clean funk guitar, muted stabs on the off-beats"
    bars = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    style, notes = generate(analysis, prompt, bars=bars, seed=seed)
    base = "/tmp/take"
    midi_path = export_midi(notes, base + ".mid", analysis["tempo"])
    wav_path, dur = render_wav(notes, base + ".wav", analysis["tempo"], prompt=prompt)
    print(json.dumps({"style": style, "notes": len(notes), "bars": bars,
                      "midi": midi_path, "wav": wav_path, "seconds": round(dur, 2)}))
