import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_vectors  # noqa: E402

V1 = Path(__file__).resolve().parents[1] / "vectors" / "v1"
SPEC = Path(__file__).resolve().parents[1] / "SPEC.md"
WITHHELD_ERRATUM = (
    Path(__file__).resolve().parents[1]
    / "docs" / "upstream-errata"
    / "2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md"
)

EXPECTED_IDS = {
    "qwen2.5-coder-7b-instruct-q8_0", "qwen2.5-coder-14b-instruct-q4_K_M",
    "qwen2.5-coder-1.5b-instruct-q8_0", "codegemma-7b-instruct-q8_0",
    "gemma2-9b", "mistral-nemo-latest",
    "deepseek-coder-v2-16b-lite-instruct-q5_K_M",
    "gemma-4-12b-it-qat-q4_0-latest",
    "qwen3.6-35b-a3b-reap48-ours-q4km",
    "qwen3.6-35b-a3b-reap48-mtp-trap",
}

# The ninth planned E1-sweep vector, `qwen3.8-27b`, is deliberately NOT in v1 — see
# test_qwen38_27b_is_deliberately_absent for the reason and the evidence,
# and WITHHELD_ERRATUM for the filed record. Ruled 2026-08-27: a documented
# v1 scope reduction (design spec section 4, plan Task 4 item 7).
WITHHELD_IDS = {"qwen3.8-27b"}


def load(vid):
    return json.loads((V1 / f"{vid}.json").read_text())


def test_v1_validates_clean():
    assert validate_vectors.validate_dir(V1) == []


def test_all_expected_ids_present():
    present = {p.stem for p in V1.glob("*.json")} - {"MANIFEST"}
    assert EXPECTED_IDS <= present


def test_anchor_values_pinned():
    assert load("qwen2.5-coder-7b-instruct-q8_0")["expected"]["kv_bytes_per_token"] == 57344
    assert load("qwen2.5-coder-14b-instruct-q4_K_M")["expected"]["kv_bytes_per_token"] == 196608
    assert load("deepseek-coder-v2-16b-lite-instruct-q5_K_M")["expected"]["experts"] == {"count": 64, "used": 6}
    assert load("gemma-4-12b-it-qat-q4_0-latest")["expected"] == {"refuses": True}


def test_hybrid_anchor_values():
    """The R3/R4 case: charge the 10 attention layers, not the 40 blocks.

    Every number here is bloomery's, measured on this exact GGUF: kv 20,480
    B/tok and recurrent_state_bytes 65,863,680 B were read from `/status` on
    two identical boots (2026-08-22-g5v4-reap48-baselines.md §7), and the
    2026-08-21 spike caught llama.cpp allocating `1070.00 MiB / 54,784 cells`
    = exactly 20,480 B/token while the daemon charged 81,920 — a measured
    4.00x over-charge. 81,920 is pinned as must_not_equal because it is the
    number a conforming implementation must NOT produce.
    """
    v = load("qwen3.6-35b-a3b-reap48-ours-q4km")
    assert v["expected"]["kv_bytes_per_token"] == 20480
    assert v["expected"]["attention_layers"] == 10
    assert v["expected"]["recurrent_state_bytes"] == 65863680
    assert v["expected"]["serving_block_count"] == 40
    assert 81920 in v["must_not_equal"]["kv_bytes_per_token"]
    assert v["source"]["gguf_sha256"].startswith("90e2181e")


def test_mtp_trap_pins_serving_block_count():
    """R6: an MTP layer is not a serving layer.

    The pre-patch REAP-48-ours conversion emitted block_count 41 alongside
    `qwen35moe.nextn_predict_layers 1` while writing only 40 blocks of
    tensors, and the GGUF would not load (`missing tensor
    'blk.40.attn_norm.weight'`). The key name is the one the artifact
    actually carried — the HF-config spelling `mtp_num_hidden_layers` never
    reaches the GGUF — so the metadata key is pinned by name as well as by
    value; renaming it in the vector would stop exercising R6.
    """
    v = load("qwen3.6-35b-a3b-reap48-mtp-trap")
    assert v["metadata"]["qwen35moe.nextn_predict_layers"] == 1
    assert v["metadata"]["qwen35moe.block_count"] == 41
    assert v["expected"]["serving_block_count"] == 40
    assert 41 in v["must_not_equal"]["serving_block_count"]


def test_spec_r6_names_the_key_the_trap_vector_carries():
    """R6 must name the GGUF spelling, or its own vector is unpassable.

    `mtp_num_hidden_layers` is HF `config.json`'s name for the quantity;
    what `convert_hf_to_gguf` writes — and therefore what an R6
    implementation reads — is `<arch>.nextn_predict_layers` (bloomery
    2026-08-22-reap48-ours-prune-and-acceptance.md §7 KV diff). An
    implementation applying R6 literally to the HF spelling finds no MTP
    key in qwen3.6-35b-a3b-reap48-mtp-trap, reports serving_block_count
    41, and trips that vector's own must_not_equal — contract text and
    conformance vector disagreeing on the input key. So the rule must
    carry the GGUF spelling (and, to keep the mapping legible, the HF one
    it mirrors), and the vector must carry that exact key.
    """
    spec = SPEC.read_text()
    assert "## R6 — serving block count" in spec
    r6 = spec.split("## R6 — serving block count")[1].split("## R7")[0]
    assert "block_count - <arch>.nextn_predict_layers" in r6
    assert "mtp_num_hidden_layers" in r6
    trap = load("qwen3.6-35b-a3b-reap48-mtp-trap")
    mtp_keys = [k for k in trap["metadata"] if k.endswith("nextn_predict_layers")]
    assert mtp_keys == ["qwen35moe.nextn_predict_layers"]
    assert trap["metadata"][mtp_keys[0]] == 1


def test_every_affected_vector_carries_must_not_equal():
    for vid in ["codegemma-7b-instruct-q8_0", "gemma2-9b", "mistral-nemo-latest",
                "deepseek-coder-v2-16b-lite-instruct-q5_K_M"]:
        assert "must_not_equal" in load(vid), vid


def test_qwen38_27b_is_deliberately_absent():
    """qwen3.8:27b has no honest v1 vector, and must not gain one by default.

    Its e1-sweep metadata states `qwen35.full_attention_interval: 4`,
    `qwen35.nextn_predict_layers: 1` and a full set of `qwen35.ssm.*` keys,
    so SPEC.md R3/R4/R6 apply. assay's published kv for it (260 KiB/token,
    tier-enthusiast/ERRATA.md and tier-enthusiast-2026-08/qwen3.8-27b.json)
    charges all 65 blocks as attention layers — 266240 == 2*65*4*256*2 —
    where R6+R3 give (65-1)/4 = 16 attention layers, a 4.06x over-charge of
    exactly the class the design spec records against bloomery turn 5. No
    published evidence carries the R3/R4-conforming kv or its
    recurrent_state_bytes, so per design spec section 4 the case waits for a
    later set rather than entering as an unverified number. Adding it back
    with the 260 KiB figure would pin, as the contract, the very defect this
    repo exists to catch.

    Ruled 2026-08-27 as a documented v1 scope reduction; the reason is filed
    at WITHHELD_ERRATUM and recorded in design spec section 4.
    """
    present = {p.stem for p in V1.glob("*.json")} - {"MANIFEST"}
    assert present & WITHHELD_IDS == set()
    assert present == EXPECTED_IDS


def test_withheld_ids_are_documented():
    """A withheld case is only defensible while its reason stays published.

    Pins the filed erratum: it must name every withheld id, the published
    figure that must not be pinned as `expected` (266240 bytes = 260
    KiB/token), and the three rules the figure conflicts with. Deleting or
    hollowing out the record turns a ruled scope reduction back into an
    unexplained gap, so the corpus fails instead.
    """
    text = WITHHELD_ERRATUM.read_text()
    for vid in WITHHELD_IDS:
        assert vid in text, vid
    assert "266240" in text
    for rule in ("R3", "R4", "R6"):
        assert rule in text, rule


def test_windows_only_on_the_designated_vector():
    for path in V1.glob("*.json"):
        if path.name == "MANIFEST.json":
            continue
        vec = json.loads(path.read_text())
        if vec["id"] != "qwen2.5-coder-7b-instruct-q8_0":
            assert "windows" not in vec["expected"], vec["id"]
