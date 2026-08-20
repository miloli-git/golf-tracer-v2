# PRD-10 — Share-readiness

**Type:** new · **Milestone:** M4 · **Effort:** ~2–3 h (continuous + final pass) · **Owner:** `README.md`, `docs/`, `tests/test_hygiene.py`, CI, `LICENSE`, `examples/`

## Goal
Another golfer clones the repo, installs, points it at their own phone video and gets a reel. Nothing personal in the tree.

## Requirements
1. README: install (Windows/Mac), capture guide (LESSONS §20 in user terms), quickstart `golftracer reel my.mp4`, labelling walkthrough, troubleshooting (autorotation, A/V offset, abstained shots).
2. Hygiene test in CI: grep for private drive roots, workstation mounts, private-notes references and personal names/places; fail on any hit.
3. `examples/`: one short synthetic or consented clip (< 20 MB) with expected outputs, used by the CI smoke test `golftracer reel examples/clip.mp4`.
4. Clean-machine run: fresh venv on a machine/VM with no local data produces a reel from the example clip.
5. `LICENSE` MIT; `CONTRIBUTING.md` one page; tag `v2.0.0` when M4 passes.
6. Repo stays private until 1–5 pass; flipping public is a separate, explicit decision.

## Acceptance
- CI green (tests + hygiene + example smoke).
- Clean-machine reel produced and visually checked.
- LESSONS.md scrubbed of place names and personal session references.

## Non-goals
Docs site, PyPI packaging (can follow).

## Dependencies
Everything; final pass after PRD-09.
