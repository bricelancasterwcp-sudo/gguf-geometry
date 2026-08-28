"""v2: the full eleven-vector set — v1 carried forward, plus qwen3.8-27b.

Mirrors tests/test_v1_vectors.py, and adds the two pins v2 needs that v1
could not have: that the carried-forward vectors really are v1's bytes
(SPEC.md versioning — old sets are never edited, and a "carry forward"
that quietly rewrote a vector would be an edit performed one directory
over), and that the case v1 withheld entered here on live evidence rather
than on the published figure it was withheld from.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_vectors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "vectors" / "v1"
V2 = ROOT / "vectors" / "v2"
SPEC = ROOT / "SPEC.md"
README = ROOT / "README.md"
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ERRATUM = (
    ROOT / "docs" / "upstream-errata"
    / "2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md"
)

# v1's ten, plus the vector v1 withheld.
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

# The only two carried-forward files v2 is allowed to differ on, and the only
# thing either is allowed to differ *by*: metadata keys added verbatim from
# the show capture each already cites, plus the provenance sentence saying so.
# Both `expected` blocks are pinned identical to v1's below — a metadata
# completeness upgrade is not a re-derivation.
UPGRADED = {
    "gemma-4-12b-it-qat-q4_0-latest.json": [
        "gemma4.attention.key_length_swa",
        "gemma4.attention.value_length_swa",
        "gemma4.attention.sliding_window",
        "gemma4.attention.sliding_window_pattern",
    ],
    "deepseek-coder-v2-16b-lite-instruct-q5_K_M.json": [
        "deepseek2.attention.value_length",
    ],
}


def load(vid, vector_dir=V2):
    return json.loads((vector_dir / f"{vid}.json").read_text())


def manifest(vector_dir):
    return json.loads((vector_dir / "MANIFEST.json").read_text())


def test_v2_validates_clean():
    assert validate_vectors.validate_dir(V2) == []


def test_the_set_is_exactly_eleven_named_vectors():
    present = {p.stem for p in V2.glob("*.json")} - {"MANIFEST"}
    assert present == EXPECTED_IDS


def test_manifest_declares_the_set_version_and_covers_every_file():
    m = manifest(V2)
    assert m["set_version"] == "v2"
    files = {p.name for p in V2.glob("*.json")} - {"MANIFEST.json"}
    assert set(m["files"]) == files
    assert len(files) == len(EXPECTED_IDS)


def test_carried_forward_vectors_are_v1_bytes():
    """v2 carries v1 forward; it does not re-author it.

    Nine of v1's ten files must hash to exactly their v1 manifest entry —
    same bytes, same shas, comparable across sets. The two upgraded files
    are excluded here and pinned by the two upgrade tests below.
    """
    v1_files, v2_files = manifest(V1)["files"], manifest(V2)["files"]
    for name, sha in v1_files.items():
        if name in UPGRADED:
            assert v2_files[name] != sha, f"{name} claims an upgrade but is v1's bytes"
            continue
        assert v2_files[name] == sha, name


def test_gemma4_swa_upgrade_adds_inputs_without_moving_expected():
    """The R8 refusal now shows the keys its own provenance warns against.

    v1's gemma-4 vector said an implementation guessing head_count_kv "from
    the *_swa keys" violates R8 while never stating them, so the tempting
    inputs were invisible to anything reading the vector. v2 states them,
    verbatim from the same capture. The refusal is unchanged: head_count_kv
    is still null and key_length_swa is a second head width, not a KV head
    count.
    """
    name = "gemma-4-12b-it-qat-q4_0-latest"
    v1, v2 = load(name, V1), load(name)
    assert v2["expected"] == v1["expected"] == {"refuses": True}
    assert v2["verification"] == v1["verification"]
    for key in UPGRADED[f"{name}.json"]:
        assert key in v2["metadata"], key
        assert key not in v1["metadata"], key
    assert v2["metadata"]["gemma4.attention.key_length_swa"] == 256
    assert v2["metadata"]["gemma4.attention.value_length_swa"] == 256
    assert v2["metadata"]["gemma4.attention.sliding_window"] == 1024
    assert v2["metadata"]["gemma4.attention.sliding_window_pattern"] is None
    assert v2["metadata"]["gemma4.attention.head_count_kv"] is None
    # v1's keys survive the addition, values untouched.
    assert all(v2["metadata"][k] == v for k, v in v1["metadata"].items())


def test_deepseek_states_both_head_widths_without_moving_expected():
    """MLA: value width 128 != key width 192, and the vector now says so.

    R1 makes key_length authoritative and R2 charges `2 * layers *
    head_count_kv * head_dim * bytes` with that one head_dim, which is the
    arithmetic behind assay's published, hardware-verified 324 KiB/token.
    SPEC.md states no rule for value_length != key_length, so stating the
    input must not move the output: `expected` is pinned identical to v1's.
    Deriving a K/V-split figure here is banned twice over (no
    implementation; never invent a value), so the gap stays visible.
    """
    name = "deepseek-coder-v2-16b-lite-instruct-q5_K_M"
    v1, v2 = load(name, V1), load(name)
    assert v2["expected"] == v1["expected"]
    assert v2["expected"]["kv_bytes_per_token"] == 331776
    assert v2["must_not_equal"] == v1["must_not_equal"]
    assert v2["metadata"]["deepseek2.attention.key_length"] == 192
    assert v2["metadata"]["deepseek2.attention.value_length"] == 128
    assert "deepseek2.attention.value_length" not in v1["metadata"]
    assert all(v2["metadata"][k] == v for k, v in v1["metadata"].items())


def test_qwen38_27b_anchor_values():
    """The case v1 withheld, entering on the live run that closed it.

    Every term is assay's fixed instrument (master 24420f2) reading this
    blob's metadata live on 2026-08-27, verbatim in
    `docs/superpowers/evidence/qwen38-27b-live-2026-08-27/geometry-qwen3.8-27b.json`
    (committed at assay 8892d0e): kv_kib_per_token 64 = 65,536 B/token,
    attention_layer_count 16, serving_block_count 64 (65 raw blocks minus
    the one nextn layer, R6), recurrent_state_bytes 156,893,184 (R4), and
    no expert keys at all — R5's absent-is-null, not a 0-expert MoE.
    """
    v = load("qwen3.8-27b")
    assert v["expected"]["kv_bytes_per_token"] == 65536
    assert v["expected"]["attention_layers"] == 16
    assert v["expected"]["serving_block_count"] == 64
    assert v["expected"]["recurrent_state_bytes"] == 156893184
    assert v["expected"]["experts"] is None
    assert v["verification"] == "hardware-verified"
    assert v["source"]["gguf_sha256"].startswith("f5f1dd89")
    assert v["metadata"]["qwen35.block_count"] == 65
    assert v["metadata"]["qwen35.nextn_predict_layers"] == 1
    assert v["metadata"]["qwen35.full_attention_interval"] == 4
    assert v["metadata"]["qwen35.attention.head_count_kv"] == 4
    assert v["metadata"]["qwen35.attention.key_length"] == 256
    assert any(k.startswith("qwen35.ssm.") for k in v["metadata"])


def test_qwen38_27b_bans_the_published_over_charge():
    """Both historical wrong answers, pinned as adversarial cases.

    266240 = 2*65*4*256*2 is assay's published 260 KiB/token, charging all
    65 raw blocks as attention layers (4.0625x, R3+R6); 65 is that raw
    count. An implementation that regresses to either trips this vector by
    construction rather than by review.
    """
    v = load("qwen3.8-27b")
    assert v["must_not_equal"]["kv_bytes_per_token"] == [266240]
    assert v["must_not_equal"]["serving_block_count"] == [65]
    assert 266240 != v["expected"]["kv_bytes_per_token"]
    for key in v["must_not_equal"]:
        if key != "note":
            assert key in v["expected"], key


def test_the_withheld_case_landed_in_v2_and_v1_stays_frozen():
    """Old sets are never edited — the closure lands in the next set.

    v1's record (its erratum and `test_qwen38_27b_is_deliberately_absent`)
    stays true forever: the vector is still absent there. What changed is
    that a conforming live measurement exists, which is exactly what the
    erratum named as closing the case, so the vector enters v2.
    """
    v1_ids = {p.stem for p in V1.glob("*.json")} - {"MANIFEST"}
    assert "qwen3.8-27b" not in v1_ids
    assert "qwen3.8-27b" in {p.stem for p in V2.glob("*.json")}
    text = ERRATUM.read_text()
    assert "qwen38-27b-live-2026-08-27" in text
    assert "v2" in text


def test_spec_records_the_v2_set():
    """A set nobody wrote down is a set consumers cannot trust."""
    spec = SPEC.read_text()
    assert "vectors/v2" in spec
    assert "2026-08-27" in spec
    assert "qwen3.8-27b" in spec


def test_readme_and_ci_check_the_v2_set():
    """The mandated check must cover the current set, not only the frozen one."""
    readme = README.read_text()
    assert re.search(r"tools/validate_vectors\.py\s+vectors/v2", readme)
    workflows = sorted(WORKFLOW_DIR.glob("*.y*ml")) if WORKFLOW_DIR.is_dir() else []
    assert workflows, "no CI workflow to check"
    text = "\n".join(wf.read_text() for wf in workflows)
    assert re.search(r"tools/validate_vectors\.py\s+vectors/v2", text)


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


def test_every_affected_vector_carries_must_not_equal():
    for vid in ["codegemma-7b-instruct-q8_0", "gemma2-9b", "mistral-nemo-latest",
                "deepseek-coder-v2-16b-lite-instruct-q5_K_M", "qwen3.8-27b"]:
        assert "must_not_equal" in load(vid), vid


def test_windows_only_on_the_designated_vector():
    for path in V2.glob("*.json"):
        if path.name == "MANIFEST.json":
            continue
        vec = json.loads(path.read_text())
        if vec["id"] != "qwen2.5-coder-7b-instruct-q8_0":
            assert "windows" not in vec["expected"], vec["id"]
