import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_vectors  # noqa: E402

V1 = Path(__file__).resolve().parents[1] / "vectors" / "v1"
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
}

# The ninth planned vector, `qwen3.8-27b`, is deliberately NOT in v1 — see
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
