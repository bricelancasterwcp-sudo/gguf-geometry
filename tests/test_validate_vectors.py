import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_vectors  # noqa: E402
import gen_manifest  # noqa: E402

GOOD = {
    "id": "example-model",
    "source": {"gguf_sha256": "a" * 64, "provenance": "some evidence doc"},
    "metadata": {"general.architecture": "qwen2"},
    "expected": {"kv_bytes_per_token": 57344, "experts": None},
    "verification": "hardware-verified",
}


def write(tmp_path, name, obj):
    (tmp_path / name).write_text(json.dumps(obj))


def test_valid_vector_passes(tmp_path):
    write(tmp_path, "example-model.json", GOOD)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert validate_vectors.validate_dir(tmp_path) == []


def test_expected_without_kv_or_refuses_is_a_violation(tmp_path):
    bad = dict(GOOD, expected={"experts": None})
    write(tmp_path, "example-model.json", bad)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    violations = validate_vectors.validate_dir(tmp_path)
    assert any("kv_bytes_per_token" in v for v in violations)


def test_refuses_true_is_a_valid_expected(tmp_path):
    ok = dict(GOOD, expected={"refuses": True})
    write(tmp_path, "example-model.json", ok)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert validate_vectors.validate_dir(tmp_path) == []


def test_bad_verification_flag_is_a_violation(tmp_path):
    bad = dict(GOOD, verification="trust-me")
    write(tmp_path, "example-model.json", bad)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert any("verification" in v for v in validate_vectors.validate_dir(tmp_path))


def test_must_not_equal_requires_note(tmp_path):
    bad = dict(GOOD, must_not_equal={"kv_bytes_per_token": [81920]})
    write(tmp_path, "example-model.json", bad)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert any("note" in v for v in validate_vectors.validate_dir(tmp_path))


def test_must_not_equal_key_must_name_a_field_in_expected(tmp_path):
    """A ban on a field the vector never states is silently inert.

    `tools/check.py` compares `must_not_equal` against a consumer's emitted
    result, and a consumer's result mirrors the shape of `expected` (design
    spec section 5). So a banned field that `expected` does not name is
    `_MISSING` in every result, the ban is skipped, and the adversarial case
    passes vacuously — which is how a typo'd field name (`serving_block_counts`
    for `serving_block_count`) stops pinning anything without anyone noticing.
    The vector's own vocabulary is `expected`'s keys; a ban must speak it.
    """
    bad = dict(
        GOOD,
        must_not_equal={"attention_layers": [40], "note": "field never stated in expected"},
    )
    write(tmp_path, "example-model.json", bad)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    violations = validate_vectors.validate_dir(tmp_path)
    assert any("attention_layers" in v and "expected" in v for v in violations), violations


def test_must_not_equal_key_present_in_expected_passes(tmp_path):
    ok = dict(
        GOOD,
        expected={"kv_bytes_per_token": 57344, "experts": None},
        must_not_equal={"kv_bytes_per_token": [81920], "note": "historical wrong answer"},
    )
    write(tmp_path, "example-model.json", ok)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert validate_vectors.validate_dir(tmp_path) == []


def test_manifest_sha_mismatch_is_a_violation(tmp_path):
    write(tmp_path, "example-model.json", GOOD)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    write(tmp_path, "example-model.json", dict(GOOD, id="tampered"))
    assert any("sha" in v.lower() for v in validate_vectors.validate_dir(tmp_path))


def test_missing_manifest_is_a_violation(tmp_path):
    write(tmp_path, "example-model.json", GOOD)
    assert any("MANIFEST" in v for v in validate_vectors.validate_dir(tmp_path))


def test_source_needs_sha_or_reason(tmp_path):
    bad = dict(GOOD, source={"provenance": "doc"})
    write(tmp_path, "example-model.json", bad)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert any("gguf_sha256" in v for v in validate_vectors.validate_dir(tmp_path))


def test_regenerating_manifest_does_not_hash_itself(tmp_path):
    write(tmp_path, "example-model.json", GOOD)
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    gen_manifest.write_manifest(tmp_path, set_version="v1")
    assert validate_vectors.validate_dir(tmp_path) == []
