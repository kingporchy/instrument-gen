"""Instrument Generator — take CLI.
Analyze a track (DAW-style with host hints, or standalone), generate an
instrument take (MIDI + audio render).

Usage:
  python core/generate_take.py TRACK [options]

Examples:
  python core/generate_take.py bounce.wav --bpm 102 --bar 0 \
      --prompt "warm clean funk guitar, muted stabs on the off-beats"
  python core/generate_take.py loop.aiff --bars 8 --seed 11
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import generator as G


def main():
    ap = argparse.ArgumentParser(description="Analyze a track and generate an instrument take")
    ap.add_argument("audio", help="path to .wav/.aiff/.mp3/.flac track")
    ap.add_argument("--bpm", type=float, default=None, help="host tempo hint (DAW path)")
    ap.add_argument("--bar", type=int, default=None, help="0..3: bar-beat index of clip start (DAW path)")
    ap.add_argument("--from", dest="t_from", type=float, default=None, help="analyze from (seconds)")
    ap.add_argument("--to", dest="t_to", type=float, default=None, help="analyze to (seconds)")
    ap.add_argument("--prompt", default="warm clean funk guitar, muted stabs on the off-beats")
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="output base path (default takes/<audioname>-take<seed>)")
    args = ap.parse_args()

    analysis = A.analyze(args.audio, host_bpm=args.bpm, t_from=args.t_from,
                         t_to=args.t_to, bar_beat_index=args.bar)
    style, notes = G.generate(analysis, args.prompt, bars=args.bars, seed=args.seed)

    base = args.out or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "takes")),
        f"{os.path.splitext(os.path.basename(args.audio))[0]}-take{args.seed}")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    midi_path = G.export_midi(notes, base + ".mid", analysis["tempo"])
    sf2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "sf2", "FluidR3_GM.sf2")
    rendered_with = "synth (additive fallback)"
    try:
        wav_path, dur = G.render_via_fluidsynth(midi_path, base + ".wav", os.path.abspath(sf2))
        rendered_with = "FluidSynth + FluidR3 GM"
    except Exception as e:
        wav_path, dur = G.render_wav(notes, base + ".wav", analysis["tempo"], prompt=args.prompt)
        render_note = str(e)[:120] if False else None
    summary_note = rendered_with

    summary = {
        "style": style, "prompt": args.prompt, "notes": len(notes),
        "bars": args.bars, "seconds": round(dur, 2), "seed": args.seed,
        "sourceAnalysis": {k: analysis[k] for k in
                           ("tempo", "key", "tempoConfidence", "keyConfidence")},
        "midi": midi_path, "wav": wav_path, "renderer": summary_note,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
