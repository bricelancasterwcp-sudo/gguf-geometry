"""The two commands that check this repo are pinned, not folklore.

A contract repo is only as good as the check that runs over it. This module
pins the *shape* of that check in two places so they cannot drift:

* README documents it, so a consumer vendoring `vectors/v1/` can reproduce it.
* Any workflow under `.github/workflows/` must run the same two commands.

The workflow file itself is a separate deliverable (plan Task 6,
`.superpowers/sdd/2026-08-27-gguf-geometry/task-6-brief.md`); until it lands
the second test skips loudly rather than passing vacuously. It needs no edit
when it lands — it starts enforcing on its own.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
WORKFLOW_DIR = ROOT / ".github" / "workflows"

# The mandated check, by shape rather than by exact string: the repo's own
# runbook prefixes PYTHONDONTWRITEBYTECODE=1 and says `python3`, CI's
# setup-python says `python`. What must not vary is which suite and which
# vector directory get run.
MANDATED = {
    "pytest over tests/": lambda cmd: "-m pytest" in cmd and "tests/" in cmd,
    "validate_vectors.py over vectors/v1": (
        lambda cmd: "tools/validate_vectors.py" in cmd and "vectors/v1" in cmd
    ),
}

# SPEC.md / README state stdlib-only Python 3.12+; a CI that pins an older
# interpreter would be checking a Python this repo does not claim to support.
MIN_PYTHON = (3, 12)

_BLOCK_SCALARS = {"|", "|-", "|+", ">", ">-", ">+"}


def _fenced_commands(markdown: str) -> list[str]:
    """Every command line inside README's shell code fences."""
    cmds: list[str] = []
    for block in re.findall(r"^```(?:bash|sh|console)?\n(.*?)^```", markdown, re.M | re.S):
        for line in block.splitlines():
            line = line.strip().lstrip("$ ").strip()
            if line and not line.startswith("#"):
                cmds.append(line)
    return cmds


def _run_commands(workflow: str) -> list[str]:
    """Every command a workflow's `run:` steps execute (inline or block scalar)."""
    cmds: list[str] = []
    lines = workflow.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)(?:-\s*)?run:\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        indent, rest = len(m.group(1)), m.group(2).strip()
        if rest not in _BLOCK_SCALARS:
            cmds.append(rest)
            i += 1
            continue
        i += 1
        while i < len(lines):
            body = lines[i]
            if body.strip() and (len(body) - len(body.lstrip())) <= indent:
                break
            if body.strip():
                cmds.append(body.strip())
            i += 1
    return cmds


def _pinned_pythons(workflow: str) -> list[tuple[int, ...]]:
    versions: list[tuple[int, ...]] = []
    for line in workflow.splitlines():
        if "python-version" not in line:
            continue
        for found in re.findall(r"(\d+)\.(\d+)", line.split("python-version", 1)[1]):
            versions.append((int(found[0]), int(found[1])))
    return versions


def test_readme_documents_the_mandated_check_commands():
    """A vendored vector set is worthless if nobody can rerun its check."""
    cmds = _fenced_commands(README.read_text())
    for name, matches in MANDATED.items():
        assert any(matches(c) for c in cmds), f"README documents no command running {name}: {cmds}"


def test_ci_workflow_runs_the_mandated_check_commands():
    """CI must run the documented check — not a subset, not a different corpus.

    Skipped, not silently passed, while the workflow is unwritten: a green
    test that asserted nothing would be exactly the false assurance this
    repo exists to prevent.
    """
    workflows = sorted(WORKFLOW_DIR.glob("*.y*ml")) if WORKFLOW_DIR.is_dir() else []
    if not workflows:
        pytest.skip(
            "no .github/workflows/*.yml yet — the workflow is plan Task 6's deliverable "
            "(.superpowers/sdd/2026-08-27-gguf-geometry/task-6-brief.md); this assertion "
            "starts enforcing the moment that file lands, with no edit here"
        )
    cmds = [c for wf in workflows for c in _run_commands(wf.read_text())]
    for name, matches in MANDATED.items():
        assert any(matches(c) for c in cmds), (
            f"no workflow step runs {name}; steps found: {cmds}"
        )
    for version in _pinned_pythons("\n".join(wf.read_text() for wf in workflows)):
        assert version >= MIN_PYTHON, (
            f"CI pins python {'.'.join(map(str, version))}; this repo states "
            f"{'.'.join(map(str, MIN_PYTHON))}+"
        )
