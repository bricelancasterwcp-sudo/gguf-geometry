"""Compare an implementation's results against vectors/<set>/*.json.

Stdlib only. This module never derives or checks a geometry value — it
loads vectors (skipping MANIFEST.json) and compares them field-by-field
against a results file a consumer emits. See SPEC.md and
docs/superpowers/specs/2026-08-27-gguf-geometry-design.md section 5.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_MISSING = object()


@dataclass(frozen=True)
class Report:
    passed: list[str]
    failed: list[tuple[str, str]]
    uncovered: list[str]


def _load_vectors(vector_dir: Path) -> list[dict]:
    vector_dir = Path(vector_dir)
    paths = sorted(p for p in vector_dir.glob("*.json") if p.name != "MANIFEST.json")
    return [json.loads(p.read_text()) for p in paths]


def _trim_windows(windows: list[dict]) -> list[dict]:
    """Vector window scenarios carry `terms` (inputs); results don't echo
    them back, so compare only the outputs, positionally (list equality
    is already index-by-index)."""
    return [
        {"usable_window": w["usable_window"], "limited_by": w["limited_by"]}
        for w in windows
    ]


def _field_diffs(expected: dict, result: dict) -> list[str]:
    diffs = []
    for key, expected_value in expected.items():
        if key == "windows":
            expected_value = _trim_windows(expected_value)
        actual = result.get(key, _MISSING)
        if actual is _MISSING:
            diffs.append(f"{key}: expected {expected_value}, got missing")
        elif actual != expected_value:
            diffs.append(f"{key}: expected {expected_value}, got {actual}")
    return diffs


def _must_not_equal_diffs(must_not_equal: dict, result: dict) -> list[str]:
    diffs = []
    note = must_not_equal.get("note", "")
    for field_name, banned_values in must_not_equal.items():
        if field_name == "note":
            continue
        actual = result.get(field_name, _MISSING)
        if actual is _MISSING:
            continue
        if actual in banned_values:
            diffs.append(f"{field_name}: got banned value {actual} ({note})")
    return diffs


def compare(results: dict, vector_dir) -> Report:
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    uncovered: list[str] = []

    for vector in _load_vectors(Path(vector_dir)):
        vector_id = vector["id"]
        if vector_id not in results:
            uncovered.append(vector_id)
            continue

        result = results[vector_id]
        diffs = _field_diffs(vector.get("expected", {}), result)
        must_not_equal = vector.get("must_not_equal")
        if isinstance(must_not_equal, dict):
            diffs.extend(_must_not_equal_diffs(must_not_equal, result))

        if diffs:
            failed.extend((vector_id, diff) for diff in diffs)
        else:
            passed.append(vector_id)

    return Report(passed=passed, failed=failed, uncovered=uncovered)


def _print_report(report: Report) -> None:
    for vector_id in report.passed:
        print(f"PASS {vector_id}")
    for vector_id, diff in report.failed:
        print(f"FAIL {vector_id}: {diff}")
    for vector_id in report.uncovered:
        print(f"UNCOVERED {vector_id}")
    failed_ids = {vector_id for vector_id, _ in report.failed}
    print(
        f"{len(report.passed)} passed, {len(failed_ids)} failed, "
        f"{len(report.uncovered)} uncovered"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check.py <results.json> <vector-dir>", file=sys.stderr)
        return 2

    results_path = Path(argv[1])
    try:
        results = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read results file ({exc})", file=sys.stderr)
        return 2
    if not isinstance(results, dict):
        print("error: results file must be a JSON object", file=sys.stderr)
        return 2

    report = compare(results, argv[2])
    _print_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
