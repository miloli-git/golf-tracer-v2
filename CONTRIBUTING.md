# Contributing

AI-agent contributions follow the same rules; `AGENTS.md` is the operational guide and its invariants are binding.

Small, focused PRs against `master`. Before opening one:

1. **Run the tests.** `pip install -e ".[dev]"` then `pytest`. All tests must pass; the hygiene test (`tests/test_hygiene.py`) fails on private paths or personal names — keep them out of code, docs, and comments.
2. **No personal data.** Footage, label sets, launch-monitor exports, and trained-run artifacts never land in this repo (see `V2-SCOPE.md`, "Two repos, two jobs"). Test fixtures must be synthetic (`tests/fixtures/make_fixture.py`) or short consented clips.
3. **Determinism matters.** The compositor and trackers are deterministic by design; avoid wall-clock, RNG without a fixed seed, and platform-dependent decode paths (see `LESSONS.md`).
4. **Behavioural changes need evidence.** If a change affects tracking or rendering output, include before/after numbers (tests, or a QA strip description) in the PR body. Overlay parity with the v1 renderer is guarded by golden tests; do not change `v1_style()` semantics without updating them.
5. **File an issue first** for anything larger than a bugfix — the PRD set in `docs/prds/` defines scope; work outside it is usually out of scope for this repo.

Bug reports: include the exact command, the source-video properties (`ffprobe` output), and the log lines around the failure. Footage itself is usually not needed — the per-stage JSON outputs are.
