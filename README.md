# Instrument Generator — engine prototype

Working prototype of the Instrument Generator spec (Phase 2 analyzer + Phase 3
MIDI generator + render), built outside the DAW first. Later wrapped by the
JUCE plugin (Phase 1) behind the same interface.

## What's here
- `core/analyze.py` — track analysis: tempo + beat grid (low-band anchored),
  downbeat, key (Krumhansl + chord-grammar refinement), chord timeline
  (Viterbi + bass-root evidence + key-diatonic post-pass), groove profile
  (16th-slot bins, microtiming ms, swing ratio). Emits MidiJson.
- `core/generator.py` — prompt → style preset → notes constrained to key +
  current chord, rhythm from groove weights, microtiming applied; exports
  MIDI (PPQ 480) and renders audio (additive synth — placeholder for samples).
- `core/generate_take.py` — CLI: one command → analysis → take.
- `tests/mktest.py` — synthetic ground-truth track; the analyzer grades
  against it (tempo 102, F# minor, F#m/Dmaj7/Bm/C#m loop, straight swing).

## Status
- Truth test (DAW path with host tempo + bar anchor): **PASS** on all metrics.
- Standalone (no host info): tempo/key/chords pass, bar-phase may shift one
  beat (documented; the plugin supplies the host bar grid, which is the
  deployment anyway).
- Audio quality is diagnostic (additive sine stack) — not ship quality.

## Run it
```bash
source ../instrument-gen-venv/bin/activate
python core/generate_take.py track.wav --bpm 102 --bar 0 \
    --prompt "warm clean funk guitar, muted stabs on the off-beats"
```
Outputs land in `takes/` (.mid for Logic, .wav to audition).

## Next
1. Real instrument rendering (sample-based or SFZ engine)
2. JUCE wrapper (Phase 1): nTracks bounce + host time/tempo/signature feed
3. Tests-as-data: spec test cases → analyzer grading harness
