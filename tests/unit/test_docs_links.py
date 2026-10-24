"""Every relative link and image in the README and the course must resolve."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "flags.md",
    *sorted((ROOT / "docs" / "lessons").glob("*.md")),
]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def relative_targets(md: Path) -> list[str]:
    targets = []
    for target in LINK.findall(md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target.split("#", 1)[0])
    return targets


@pytest.mark.parametrize("md", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(md: Path) -> None:
    missing = [t for t in relative_targets(md) if not (md.parent / t).exists()]
    assert not missing, f"{md.name} links to missing files: {missing}"
