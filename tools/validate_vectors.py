"""Validate vectors/<set>/*.json against the schema in spec section 3.

Stdlib only. This module checks *shape* (types, required fields, manifest
consistency) — it never derives or checks a geometry value. See SPEC.md
and docs/superpowers/specs/2026-08-27-gguf-geometry-design.md section 3.
"""
import hashlib
import json
import sys
from pathlib import Path

VERIFICATION_VALUES = {"hardware-verified", "derived"}
POSITIVE_INT_FIELDS = ("attention_layers", "recurrent_state_bytes", "serving_block_count")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value) -> bool:
    return _is_int(value) and value > 0


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_sha256_hex(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _check_source(prefix: str, source: dict) -> list[str]:
    violations = []
    has_sha = "gguf_sha256" in source
    has_reason = "sha_unavailable_reason" in source
    if has_sha:
        if not _is_sha256_hex(source["gguf_sha256"]):
            violations.append(
                f"{prefix}: source.gguf_sha256 must be 64 lowercase hex characters"
            )
    elif has_reason:
        if not _is_nonempty_str(source["sha_unavailable_reason"]):
            violations.append(
                f"{prefix}: source.sha_unavailable_reason must be a nonempty string"
            )
    else:
        violations.append(
            f"{prefix}: source must have gguf_sha256 or sha_unavailable_reason"
        )
    if not _is_nonempty_str(source.get("provenance")):
        violations.append(f"{prefix}: source.provenance must be a nonempty string")
    return violations


def _check_experts(prefix: str, experts) -> list[str]:
    if experts is None:
        return []
    if not isinstance(experts, dict):
        return [f"{prefix}: expected.experts must be null or an object"]
    violations = []
    for field in ("count", "used"):
        if not _is_int(experts.get(field)):
            violations.append(f"{prefix}: expected.experts.{field} must be an int")
    return violations


def _check_windows(prefix: str, windows) -> list[str]:
    if not isinstance(windows, list):
        return [f"{prefix}: expected.windows must be a list"]
    violations = []
    for i, entry in enumerate(windows):
        if not isinstance(entry, dict):
            violations.append(f"{prefix}: expected.windows[{i}] must be an object")
            continue
        if not isinstance(entry.get("terms"), dict):
            violations.append(f"{prefix}: expected.windows[{i}].terms must be an object")
        usable_window = entry.get("usable_window")
        if not _is_int(usable_window) or usable_window < 0:
            violations.append(
                f"{prefix}: expected.windows[{i}].usable_window must be an int >= 0"
            )
        if not _is_nonempty_str(entry.get("limited_by")):
            violations.append(
                f"{prefix}: expected.windows[{i}].limited_by must be a nonempty string"
            )
    return violations


def _check_expected(prefix: str, expected: dict) -> list[str]:
    violations = []
    has_kv = "kv_bytes_per_token" in expected
    if has_kv:
        if not _is_positive_int(expected["kv_bytes_per_token"]):
            violations.append(f"{prefix}: expected.kv_bytes_per_token must be a positive int")
    elif expected.get("refuses") is not True:
        violations.append(
            f"{prefix}: expected must have a positive int kv_bytes_per_token or refuses: true"
        )

    for field in POSITIVE_INT_FIELDS:
        if field in expected and not _is_positive_int(expected[field]):
            violations.append(f"{prefix}: expected.{field} must be a positive int when present")

    if "experts" in expected:
        violations.extend(_check_experts(prefix, expected["experts"]))

    if "windows" in expected:
        violations.extend(_check_windows(prefix, expected["windows"]))

    return violations


def _check_must_not_equal(prefix: str, must_not_equal) -> list[str]:
    if not isinstance(must_not_equal, dict):
        return [f"{prefix}: must_not_equal must be an object"]
    if not _is_nonempty_str(must_not_equal.get("note")):
        return [f"{prefix}: must_not_equal.note must be a nonempty string"]
    return []


def _validate_vector(prefix: str, data: dict) -> list[str]:
    violations = []

    if not _is_nonempty_str(data.get("id")):
        violations.append(f"{prefix}: id must be a nonempty string")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        violations.append(f"{prefix}: metadata must be a nonempty object")

    verification = data.get("verification")
    if verification not in VERIFICATION_VALUES:
        violations.append(
            f"{prefix}: verification must be one of {sorted(VERIFICATION_VALUES)} "
            f"(got {verification!r})"
        )

    source = data.get("source")
    if not isinstance(source, dict):
        violations.append(f"{prefix}: source must be an object")
    else:
        violations.extend(_check_source(prefix, source))

    expected = data.get("expected")
    if not isinstance(expected, dict):
        violations.append(f"{prefix}: expected must be an object")
    else:
        violations.extend(_check_expected(prefix, expected))

    if "must_not_equal" in data:
        violations.extend(_check_must_not_equal(prefix, data["must_not_equal"]))

    return violations


def _validate_vector_file(path: Path) -> list[str]:
    prefix = path.name
    try:
        text = path.read_text()
    except OSError as exc:
        return [f"{prefix}: could not read file ({exc})"]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"{prefix}: invalid JSON ({exc})"]
    if not isinstance(data, dict):
        return [f"{prefix}: top-level value must be an object"]
    return _validate_vector(prefix, data)


def _validate_manifest(manifest_path: Path, vector_files: list[Path]) -> list[str]:
    if not manifest_path.exists():
        return ["MANIFEST.json: MANIFEST.json is missing"]

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"MANIFEST.json: invalid JSON ({exc})"]
    if not isinstance(manifest, dict):
        return ["MANIFEST.json: top-level value must be an object"]

    files_field = manifest.get("files")
    if not isinstance(files_field, dict):
        return ["MANIFEST.json: files must be an object"]

    violations = []
    vector_names = {p.name for p in vector_files}
    manifest_names = set(files_field.keys())

    for name in sorted(vector_names - manifest_names):
        violations.append(f"MANIFEST.json: missing entry for {name}")
    for name in sorted(manifest_names - vector_names):
        violations.append(f"MANIFEST.json: lists {name} which is not a vector file")

    for path in vector_files:
        if path.name not in files_field:
            continue
        expected_sha = files_field[path.name]
        if not isinstance(expected_sha, str):
            violations.append(f"{path.name}: MANIFEST.json entry for {path.name} is not a string")
            continue
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            violations.append(f"{path.name}: sha256 does not match MANIFEST.json")

    return violations


def validate_dir(path) -> list[str]:
    vector_dir = Path(path)
    vector_files = sorted(p for p in vector_dir.glob("*.json") if p.name != "MANIFEST.json")

    violations = _validate_manifest(vector_dir / "MANIFEST.json", vector_files)
    for file_path in vector_files:
        violations.extend(_validate_vector_file(file_path))
    return violations


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_vectors.py <vector-dir>", file=sys.stderr)
        return 2
    violations = validate_dir(argv[1])
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
