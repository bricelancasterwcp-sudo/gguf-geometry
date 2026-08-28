"""v3: v2 carried forward, plus the MLA KV rule pinned by measurement.

Mirrors tests/test_v2_vectors.py. v3 does not add a model — it re-pins the
one vector v2 left an open gap on. R2 was specified for `key_length` alone;
SPEC.md's "Known open gap — R2 and MLA" (recorded 2026-08-27) named the
deepseek-coder-v2 vector as the case it did not cover. A live measurement
(assay `docs/superpowers/evidence/mla-kv-2026-08-27/`, verdict H-b) closed
it 2026-08-28: SPEC.md gained R9, and this set re-pins the vector to the
measured figure, 276480 B/token, disproving the three other candidates the
pre-registered protocol banded.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_vectors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "vectors" / "v2"
V3 = ROOT / "vectors" / "v3"
SPEC = ROOT / "SPEC.md"
README = ROOT / "README.md"
ERRATUM = (
    ROOT / "docs" / "upstream-errata"
    / "2026-08-27-assay-deepseek-mla-kv-overcharge.md"
)

# v2's eleven, unchanged — v3 re-pins a vector, it does not add or remove one.
EXPECTED_IDS = {
    "qwen2.5-coder-7b-instruct-q8_0", "qwen2.5-coder-14b-instruct-q4_K_M",
    "qwen2.5-coder-1.5b-instruct-q8_0", "codegemma-7b-instruct-q8_0",
    "gemma2-9b", "mistral-nemo-latest",
    "deepseek-coder-v2-16b-lite-instruct-q5_K_M",
    "gemma-4-12b-it-qat-q4_0-latest",
    "qwen3.6-35b-a3b-reap48-ours-q4km",
    "qwen3.6-35b-a3b-reap48-mtp-trap",
    "qwen3.8-27b",
}

# The only file v3 is allowed to differ on, and the only thing it is
# allowed to differ *by*: the branch's metadata additions (none, for H-b —
# the value is an empty list rather than the key being absent, so the
# carried-forward loop below still names this file explicitly), the new
# `expected.kv_bytes_per_token`, the extended `must_not_equal` list, and
# the provenance sentence recording the re-pin. Every other block
# (metadata, id, source.gguf_sha256) is unchanged from v2.
UPGRADED = {
    "deepseek-coder-v2-16b-lite-instruct-q5_K_M.json": [],
}

# The disproved-candidate note, appended verbatim to the v2 note (never
# rewritten) — see SPEC.md R9 and the erratum this cites.
NOTE_APPEND = (
    "; MLA candidates disproved by measurement (assay mla-kv-2026-08-27): "
    "331776 (k-width for both K and V — the v1/v2 pin), 31104 (MLA latent), "
    "62208 (MLA latent, transposed copy) — see SPEC R9 and "
    "docs/upstream-errata/2026-08-27-assay-deepseek-mla-kv-overcharge.md"
)


def load(vid, vector_dir=V3):
    return json.loads((vector_dir / f"{vid}.json").read_text())


def manifest(vector_dir):
    return json.loads((vector_dir / "MANIFEST.json").read_text())


def test_v3_validates_clean():
    assert validate_vectors.validate_dir(V3) == []


def test_the_set_is_exactly_eleven_named_vectors():
    present = {p.stem for p in V3.glob("*.json")} - {"MANIFEST"}
    assert present == EXPECTED_IDS


def test_manifest_declares_the_set_version_and_covers_every_file():
    m = manifest(V3)
    assert m["set_version"] == "v3"
    files = {p.name for p in V3.glob("*.json")} - {"MANIFEST.json"}
    assert set(m["files"]) == files
    assert len(files) == len(EXPECTED_IDS)


def test_carried_forward_vectors_are_v2_bytes():
    """v3 carries v2 forward; it re-pins one vector, it does not re-author the set.

    Ten of v2's eleven files must hash to exactly their v2 manifest entry —
    same bytes, same shas, comparable across sets. The re-pinned file is
    excluded here and pinned by the re-pin test below.
    """
    v2_files, v3_files = manifest(V2)["files"], manifest(V3)["files"]
    for name, sha in v2_files.items():
        if name in UPGRADED:
            assert v3_files[name] != sha, f"{name} claims a re-pin but is v2's bytes"
            continue
        assert v3_files[name] == sha, name


def test_deepseek_repinned_to_the_measured_mla_figure():
    """The gap v2 recorded and did not close, closed by measurement.

    v1/v2 pinned 331776 — R2 arithmetic (`2 * layers * head_count_kv *
    key_length * bytes`) run on a model whose metadata also states a
    different `value_length`, which R2 was never specified to cover. A
    live measurement under ollama 0.32.13 (verdict H-b) shows the runtime
    allocates separate K and V widths, 276480 B/token — SPEC.md's new R9.
    All three other pre-registered candidates the protocol banded are
    pinned as disproved: 331776 (the old R2-arithmetic pin), 31104 and
    62208 (the two latent-cache candidates a naive "it's MLA" assumption
    would have produced).
    """
    name = "deepseek-coder-v2-16b-lite-instruct-q5_K_M"
    v2, v3 = load(name, V2), load(name, V3)

    # metadata, id, and gguf_sha256 are untouched — H-b adds no metadata.
    assert v3["metadata"] == v2["metadata"]
    assert v3["id"] == v2["id"]
    assert v3["source"]["gguf_sha256"] == v2["source"]["gguf_sha256"]

    # expected: kv_bytes_per_token moves to the measured figure; everything
    # else in `expected` (the expert fields) is untouched.
    assert v3["expected"]["kv_bytes_per_token"] == 276480
    assert v2["expected"]["kv_bytes_per_token"] == 331776  # v2 stays frozen
    assert v3["expected"]["experts"] == v2["expected"]["experts"] == {"count": 64, "used": 6}

    # verification stays hardware-verified in both — v3's provenance is what
    # says this is the first time the figure is an OBSERVED allocation.
    assert v3["verification"] == v2["verification"] == "hardware-verified"

    # must_not_equal: the v1/v2 pin (221184) stays, the branch adds three.
    assert v3["must_not_equal"]["kv_bytes_per_token"] == [221184, 331776, 31104, 62208]
    for candidate in (221184, 331776, 31104, 62208):
        assert candidate != v3["expected"]["kv_bytes_per_token"]

    # note: appended, never rewritten.
    assert v3["must_not_equal"]["note"] == v2["must_not_equal"]["note"] + NOTE_APPEND
    assert v3["must_not_equal"]["note"].startswith(v2["must_not_equal"]["note"])

    # provenance: appended, never rewritten, and states the v3 re-pin.
    assert v3["source"]["provenance"].startswith(v2["source"]["provenance"])
    appended = v3["source"]["provenance"][len(v2["source"]["provenance"]):]
    assert "v3 (2026-08-28)" in appended
    assert "R9" in appended
    assert "276480" in appended
    assert "mla-kv-2026-08-27" in appended
    assert re.search(r"\bOBSERVED\b", appended), "must say the kv figure is, for the first time, an observed allocation"


def test_deepseek_must_not_equal_matches_expected_field():
    """The adversarial block's own guard: it must name a field expected states."""
    v3 = load("deepseek-coder-v2-16b-lite-instruct-q5_K_M")
    for field in v3["must_not_equal"]:
        if field == "note":
            continue
        assert field in v3["expected"], field


def test_spec_records_the_v3_closure():
    """A closure nobody wrote down is a closure consumers cannot trust."""
    spec = SPEC.read_text()
    assert "vectors/v3" in spec
    assert "2026-08-28" in spec
    assert "R9" in spec


def test_erratum_names_v3_as_where_the_repin_lands():
    text = ERRATUM.read_text()
    assert "vectors/v3" in text
    assert "276,480" in text or "276480" in text


def test_readme_states_nine_rules():
    """SPEC.md gained R9 (Task 4); README's rule count must not undercount it."""
    readme = README.read_text()
    assert "nine rules" in readme
    assert re.search(r"R1[–-]R9", readme)
    assert "eight rules" not in readme


def test_anchor_values_pinned():
    assert load("qwen2.5-coder-7b-instruct-q8_0")["expected"]["kv_bytes_per_token"] == 57344
    assert load("qwen2.5-coder-14b-instruct-q4_K_M")["expected"]["kv_bytes_per_token"] == 196608
    assert load("deepseek-coder-v2-16b-lite-instruct-q5_K_M")["expected"]["experts"] == {"count": 64, "used": 6}
    assert load("gemma-4-12b-it-qat-q4_0-latest")["expected"] == {"refuses": True}


def test_hybrid_and_mtp_anchor_values_carry_forward():
    hybrid = load("qwen3.6-35b-a3b-reap48-ours-q4km")
    assert hybrid["expected"]["kv_bytes_per_token"] == 20480
    assert hybrid["expected"]["attention_layers"] == 10
    assert hybrid["expected"]["recurrent_state_bytes"] == 65863680
    assert hybrid["expected"]["serving_block_count"] == 40
    assert 81920 in hybrid["must_not_equal"]["kv_bytes_per_token"]
    trap = load("qwen3.6-35b-a3b-reap48-mtp-trap")
    assert trap["metadata"]["qwen35moe.block_count"] == 41
    assert trap["expected"]["serving_block_count"] == 40
    assert 41 in trap["must_not_equal"]["serving_block_count"]


def test_qwen38_27b_carries_forward_unchanged():
    v2, v3 = load("qwen3.8-27b", V2), load("qwen3.8-27b", V3)
    assert v3 == v2


def test_every_affected_vector_carries_must_not_equal():
    for vid in ["codegemma-7b-instruct-q8_0", "gemma2-9b", "mistral-nemo-latest",
                "deepseek-coder-v2-16b-lite-instruct-q5_K_M", "qwen3.8-27b"]:
        assert "must_not_equal" in load(vid), vid


def test_windows_only_on_the_designated_vector():
    for path in V3.glob("*.json"):
        if path.name == "MANIFEST.json":
            continue
        vec = json.loads(path.read_text())
        if vec["id"] != "qwen2.5-coder-7b-instruct-q8_0":
            assert "windows" not in vec["expected"], vec["id"]
