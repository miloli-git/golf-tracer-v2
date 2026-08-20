from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_no_private_paths_or_names() -> None:
    forbidden = [
        "H" + ":/",
        "H" + ":" + "\\",
        "F" + ":/" + "dev",
        "F" + ":" + "\\" + "dev",
        "Z" + ":/",
        "Z" + ":" + "\\",
        "/" + "Volumes",
        "192" + ".168.",
        "ob" + "sidian",
        "va" + "ult",
        "Hud" + "son",
        "Bar" + "ton",
        "Top" + "tracer",
        "mi" + "lo_label",
        "golf-shot" + "-tracer",
        "mi" + "loli-lab",
    ]
    failures: list[str] = []
    roots = [ROOT / "golftracer", ROOT / "tests", ROOT / "docs", ROOT / "tools", ROOT / "examples"]
    extra_files = [
        ROOT / "README.md", ROOT / "LESSONS.md", ROOT / "V2-SCOPE.md",
        ROOT / "AGENTS.md", ROOT / "CLAUDE.md",
        ROOT / "CONTRIBUTING.md", ROOT / "pyproject.toml",
    ]
    candidates = [
        path for root in roots for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".toml", ".yaml", ".yml"}
    ] + [path for path in extra_files if path.exists()]
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term.lower() in text.lower():
                failures.append(f"{path.relative_to(ROOT)} contains a private path/name: {term[:2]}...")
    assert not failures, "\n".join(failures)


def test_no_per_frame_seek_pattern() -> None:
    token = "-" + "ss"
    occurrences: list[tuple[Path, int]] = []
    for path in (ROOT / "golftracer").rglob("*.py"):
        count = path.read_text(encoding="utf-8").count(f'"{token}"')
        if count:
            occurrences.append((path, count))
    assert len(occurrences) == 1
    assert occurrences[0][0].name == "decode.py"
    assert occurrences[0][1] == 1

