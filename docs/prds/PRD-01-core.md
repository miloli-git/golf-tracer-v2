# PRD-01 — Core package, CLI, deterministic decode

**Type:** port+refactor · **Milestone:** M1 · **Effort:** ~3–4 h · **Owner:** `golftracer/{__init__,cli,decode,session,config}.py`, `pyproject.toml`, `tests/test_decode.py`

## Goal
A pip-installable package with one CLI and one decode path that every downstream module shares. Nothing here knows about any specific footage, path or session.

## Inputs / Outputs
- In: a video path, optional `--out DIR`, optional `--config yaml`.
- Out: `Session` object (video meta from *decoded* frames, fps, rotation), a frame-window decoder, an output directory layout: `<out>/session.json`, `<out>/impacts.json`, `<out>/swings/<k>/{tracks, labels, audit}`, `<out>/reel.mp4`.

## Requirements
1. `pyproject.toml`, package `golftracer`, console script `golftracer` with subcommands `impacts`, `track`, `label`, `render`, `reel` (later PRDs fill them; this PRD wires the parser and `reel` as a stub pipeline).
2. `decode.py`: port `tracer/swing_decode.py`. One sequential single-seek decode per window, count-indexed from window start; shared by every consumer. Per-frame ffmpeg seeks are forbidden and a test fails if a per-frame `-ss` pattern appears.
3. Dimensions come from decoded frames, never ffprobe stored size (autorotation).
4. `session.py`: `Session`, `Swing(window_start, window_end, impact_t)`, `Track` dataclasses + JSON (de)serialisation; the schema is the contract between phases and compositor.
5. `config.py`: all tunables (gates, window lengths, A/V offset default) in one dataclass with YAML override; no magic numbers in modules.
6. Zero literal personal paths. A hygiene test greps the package, tests and docs for private drive roots, workstation mounts, private-notes references and personal place names.
7. Logging to stderr with `-v`; results to files, never stdout prose.
8. Golden hook: `golftracer/golden.py` loads a manifest from `--golden PATH` or `GOLFTRACER_GOLDEN` (schema: video, impacts, club swings + labels + oracle arcs + oracle reel, ball oracle, combined oracle). `pytest -m golden` skips when unset. The manifest and everything it points to stay outside this repo (private v1 repo `golden/v2-golden.yaml`).

## Acceptance
- `pip install -e . && golftracer reel <local video> --out DIR` runs the stub pipeline to a placeholder reel with no error on Windows and Mac.
- Decode test: decoding the same 2 s window twice yields identical frame hashes; a window start rounded vs exact yields the same first frame (determinism, LESSONS §7).
- Path-grep test passes.

## Non-goals
Any tracking. GUI.

## Dependencies
None. Ports: `tracer/swing_decode.py`, `tracer/ingest.py` (metadata part only).

## Notes
LESSONS §7, §8, §25 (commit baseline before fan-out).
