import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check  # noqa: E402

VECTOR = {
    "id": "example-model",
    "source": {"gguf_sha256": "a" * 64, "provenance": "doc"},
    "metadata": {"general.architecture": "qwen2"},
    "expected": {"kv_bytes_per_token": 57344, "experts": None},
    "must_not_equal": {"kv_bytes_per_token": [114688], "note": "old head_dim formula"},
    "verification": "hardware-verified",
}


def vector_dir(tmp_path):
    (tmp_path / "example-model.json").write_text(json.dumps(VECTOR))
    return tmp_path


def test_matching_result_passes(tmp_path):
    report = check.compare(
        {"example-model": {"kv_bytes_per_token": 57344, "experts": None}},
        vector_dir(tmp_path),
    )
    assert report.passed == ["example-model"] and not report.failed


def test_wrong_kv_fails_with_diff(tmp_path):
    report = check.compare(
        {"example-model": {"kv_bytes_per_token": 999, "experts": None}},
        vector_dir(tmp_path),
    )
    (vid, diff), = report.failed
    assert vid == "example-model" and "57344" in diff and "999" in diff


def test_must_not_equal_hit_fails_naming_the_note(tmp_path):
    report = check.compare(
        {"example-model": {"kv_bytes_per_token": 114688, "experts": None}},
        vector_dir(tmp_path),
    )
    assert any("old head_dim formula" in diff for _, diff in report.failed)


def test_experts_zero_is_not_null(tmp_path):
    report = check.compare(
        {"example-model": {"kv_bytes_per_token": 57344, "experts": {"count": 0, "used": 0}}},
        vector_dir(tmp_path),
    )
    assert report.failed


def test_absent_model_is_uncovered_not_failed(tmp_path):
    report = check.compare({}, vector_dir(tmp_path))
    assert report.uncovered == ["example-model"] and not report.failed


def test_missing_expected_key_in_result_fails(tmp_path):
    report = check.compare({"example-model": {"experts": None}}, vector_dir(tmp_path))
    assert any("kv_bytes_per_token" in diff for _, diff in report.failed)


# Windows: consumed format is [{"usable_window", "limited_by"}], compared
# positionally against the vector's expected.windows scenarios (which also
# carry "terms" — the scenario inputs — that results never echo back).
WINDOW_VECTOR = dict(
    VECTOR,
    expected={
        "kv_bytes_per_token": 57344,
        "experts": None,
        "windows": [
            {"terms": {"budget_bytes": 0}, "usable_window": 0, "limited_by": "kv"},
            {"terms": {"budget_bytes": 100}, "usable_window": 4096, "limited_by": "budget"},
        ],
    },
)


def window_vector_dir(tmp_path):
    (tmp_path / "example-model.json").write_text(json.dumps(WINDOW_VECTOR))
    return tmp_path


def test_matching_windows_pass_ignoring_terms(tmp_path):
    report = check.compare(
        {
            "example-model": {
                "kv_bytes_per_token": 57344,
                "experts": None,
                "windows": [
                    {"usable_window": 0, "limited_by": "kv"},
                    {"usable_window": 4096, "limited_by": "budget"},
                ],
            }
        },
        window_vector_dir(tmp_path),
    )
    assert report.passed == ["example-model"] and not report.failed


def test_window_mismatch_at_position_fails(tmp_path):
    report = check.compare(
        {
            "example-model": {
                "kv_bytes_per_token": 57344,
                "experts": None,
                "windows": [
                    {"usable_window": 0, "limited_by": "kv"},
                    {"usable_window": 8192, "limited_by": "budget"},
                ],
            }
        },
        window_vector_dir(tmp_path),
    )
    assert any(
        "windows" in diff and "8192" in diff and "4096" in diff for _, diff in report.failed
    )


# A real vectors/v1 directory (Task 4/5) is a MANIFEST.json-bearing directory
# holding several vector files side by side. None of the tests above exercise
# that shape — each writes exactly one vector file and no MANIFEST.json. This
# pins the multi-vector, MANIFEST-present case so check.py's actual target
# shape has direct coverage rather than relying on manual smoke-testing.
def multi_vector_dir_with_manifest(tmp_path):
    (tmp_path / "model-a.json").write_text(json.dumps(dict(VECTOR, id="model-a")))
    (tmp_path / "model-b.json").write_text(
        json.dumps(
            {
                "id": "model-b",
                "source": {"gguf_sha256": "b" * 64, "provenance": "doc"},
                "metadata": {"general.architecture": "llama"},
                "expected": {"kv_bytes_per_token": 4096, "experts": None},
                "verification": "derived",
            }
        )
    )
    # Deliberately NOT vector-shaped (no "id" key) -- if check.py ever stopped
    # skipping MANIFEST.json by name, loading this file as a vector would
    # raise KeyError on vector["id"] rather than silently misbehaving.
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"set_version": "test", "files": {"model-a.json": "x", "model-b.json": "y"}})
    )
    return tmp_path


def test_manifest_json_is_skipped_in_a_multi_vector_directory(tmp_path):
    report = check.compare(
        {
            "model-a": {"kv_bytes_per_token": 57344, "experts": None},
            "model-b": {"kv_bytes_per_token": 4096, "experts": None},
        },
        multi_vector_dir_with_manifest(tmp_path),
    )
    assert sorted(report.passed) == ["model-a", "model-b"]
    assert not report.failed and not report.uncovered
