# gguf-geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A contract-only repo (SPEC.md + frozen conformance vectors) that pins GGUF metadata interpretation and the window law, plus conformance tests in assay (branch-held) and bloomery (human-gated).

**Architecture:** No geometry implementation lives here — the repo holds the written contract, sha-anchored JSON vectors whose expected values are copied from published evidence, and two stdlib tools (a vector validator and a results checker) that compute no geometry. Consumers vendor a sha-pinned copy of `vectors/v1/` and prove their own implementations against it in their own CI.

**Tech Stack:** Python 3.12+ stdlib only (tools), pytest (dev-only), GitHub Actions CI. Authoring uses `~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py` (external tool, nothing shipped).

**Spec:** `docs/superpowers/specs/2026-08-27-gguf-geometry-design.md` — read it first; the plan argues from it.

## Global Constraints

- **No geometry implementation in this repo** (spec §0). Tools validate and compare; they never derive kv/window numbers.
- **Every expected value cites published evidence** (spec §3): `source.provenance` names the evidence doc; `verification` is `"hardware-verified"` or `"derived"`. Synthetic window-law scenarios are admissible only with the arithmetic shown in the provenance string.
- **Tools are stdlib-only.** pytest is a dev dependency only.
- **Mutation discipline** (house rule): every new test module gets a mutation check — temporarily break the code under test, confirm the test fails, restore. Always with `PYTHONDONTWRITEBYTECODE=1` and `__pycache__` purged first (the stale-bytecode gotcha: a same-length mutation restored in the same second can leave the mutant's bytecode live).
- **Interference guards** (spec §6): no writes to `~/workspace/bloomery` or `~/workspace/crucible`; no cargo commands in bloomery; no GPU use anywhere in this plan. assay work happens only in a new worktree on branch `geometry-conformance` and is NEVER merged or pushed to assay master (MERGE HOLD until Brice declares the bloomery memory-battery closed).
- **Commits:** conventional format (`feat:`/`test:`/`docs:`/`chore:`), no attribution trailers.
- **Publishing is human-gated** (Task 7): no `gh repo create`, no push, before Brice's explicit go.
- Repo root for Tasks 1–7: `~/workspace/gguf-geometry` (exists; contains the committed spec).

---

### Task 1: SPEC.md, README, LICENSE

**Files:**
- Create: `SPEC.md`
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`

**Interfaces:**
- Produces: SPEC.md section numbers `R1`–`R8` (rules) that vectors and consumer tests cite by name.

- [ ] **Step 1: Write SPEC.md**

Transcribe spec §2 into a standalone contract. Structure: a short preamble (what this contract is, the no-implementation property), then rules `R1`–`R8`, each with a one-paragraph statement and, where numeric, the formula. Use exactly these rule numbers and titles:

```markdown
# The GGUF Geometry Contract (v1)

Rules R1–R8. A conforming implementation derives these values from GGUF
metadata KV pairs (the namespaced keys, e.g. `qwen2.block_count`,
`<arch>.attention.head_count_kv`) and nothing else.

## R1 — head_dim
`<arch>.attention.key_length` is authoritative when present.
Fallback `embedding_length // head_count` is admissible ONLY when
key_length is absent. (Origin: assay E1 erratum — the fallback used
unconditionally over-promised usable windows 17–33% on 6 of 23
committed profiles, both signs.)

## R2 — KV bytes per token
`2 * attention_layer_count * head_count_kv * head_dim * bytes_per_element`
(2 = K and V; bytes_per_element = 2 for f16 cache). Expert-invariant by
design: K/V heads are dense in MoE architectures.

## R3 — attention_layer_count (hybrid models)
For architectures with a `full_attention_interval` key (or equivalent):
`attention_layer_count = serving_block_count / full_attention_interval`.
Never the raw block count. (Origin: bloomery turn-5 — all-40-blocks
charged a 4× KV overcharge on Qwen3.6-35B-A3B; 10 real attention
layers.) Dense models: attention_layer_count = serving_block_count.

## R4 — recurrent state
Architectures with `ssm.*` keys charge a distinct per-context term,
`recurrent_state_bytes`, derived from the ssm dimensions. Zero ONLY
when the architecture has no recurrent layers — never as a default.

## R5 — MoE expert fields
`expert_count` / `expert_used_count` absent from metadata means the
value is absent (null/None) — NEVER 0. A dense model is not a 0-expert
MoE.

## R6 — serving block count
`serving_block_count = block_count - mtp_num_hidden_layers` when the
MTP key is present and nonzero. MTP layers are not serving layers.
(Origin: REAP-48 prune left `mtp_num_hidden_layers: 1` → 41-vs-40 →
unloadable GGUF.)

## R7 — the window law
A pure function. Named input terms: `training_ctx`, `budget_bytes`
(free accelerator bytes), `weights_bytes` (0 if already resident),
`fixed_overhead_bytes`, `kv_bytes_per_token`, `recurrent_state_bytes`,
`user_cap`. Output: `usable_window` and `limited_by` (the name of the
binding term). `usable_window = min` over the measurable candidate
terms; an unmeasured term DROPS OUT — it is never guessed. Consumers
map their own configuration onto these names.

## R8 — unknown architecture
If the keys a rule needs are absent or the architecture is unrecognized,
the implementation refuses (returns unknown/None and says why). Guessing
is a contract violation, not a fallback.

## Versioning
The vector-set directory (`vectors/v1/`, `v2/`, ...) is the contract
version. Any semantic change to a rule bumps the set. Old sets are
never edited; comparisons across sets are incomparable, not wrong.
```

- [ ] **Step 2: Write README.md**

Cover, briefly: the one-sentence purpose (one contract so the same interpretation bug can never live silently in two implementations), the no-implementation property and why (a reference implementation would be a third implementation with a fourth bug), the three lineages and their measured defects (table from spec §0), consumer model (vendor `vectors/v1/`, assert the MANIFEST sha, run your implementation over the vectors in your own CI), and pointers to SPEC.md and the tools. State plainly: expected values enter only by citation from published evidence.

- [ ] **Step 3: Write LICENSE (MIT, copyright 2026 Brice Lancaster) and .gitignore**

`.gitignore`: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.venv/`.

- [ ] **Step 4: Self-check** — every R-rule traces to a spec §2 item (8/8); no rule references a key name not shown in an example; no TBDs.

- [ ] **Step 5: Commit**

```bash
cd ~/workspace/gguf-geometry && git add -A && git commit -m "docs: SPEC.md R1-R8 contract, README, MIT license"
```

---

### Task 2: Vector validator + manifest generator

**Files:**
- Create: `tools/validate_vectors.py`
- Create: `tools/gen_manifest.py`
- Test: `tests/test_validate_vectors.py`

**Interfaces:**
- Produces: `validate_vectors.validate_dir(path) -> list[str]` (empty = valid; each string = one violation `"<file>: <problem>"`). CLI: `python3 tools/validate_vectors.py vectors/v1` exits 0 valid / 1 violations / 2 usage. `gen_manifest.py vectors/v1` writes `vectors/v1/MANIFEST.json` = `{"set_version": "v1", "files": {"<name>.json": "<sha256>", ...}}` (MANIFEST.json itself excluded).
- Consumes: nothing.

Vector schema being enforced (from spec §3):
- top-level required: `id` (str), `source` (obj), `metadata` (nonempty obj), `expected` (obj), `verification` in `{"hardware-verified","derived"}`; optional: `must_not_equal` (obj, must contain str `note`).
- `source` required: `gguf_sha256` (64 lowercase hex) OR `sha_unavailable_reason` (str — the E1 show files record daemon digests, not always raw GGUF shas); `provenance` (nonempty str).
- `expected` required: `kv_bytes_per_token` (positive int) OR `refuses: true` (the R8 case); optional: `attention_layers`, `recurrent_state_bytes`, `serving_block_count` (positive ints), `experts` (null or `{"count": int, "used": int}`), `windows` (list of `{"terms": obj, "usable_window": int, "limited_by": str}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate_vectors.py
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd ~/workspace/gguf-geometry && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_validate_vectors.py -q`
Expected: FAIL — `ModuleNotFoundError: validate_vectors`.

- [ ] **Step 3: Implement `tools/gen_manifest.py`**

```python
"""Write MANIFEST.json for a vector directory. Computes shas; no geometry."""
import hashlib
import json
import sys
from pathlib import Path


def write_manifest(vector_dir, *, set_version: str) -> Path:
    vector_dir = Path(vector_dir)
    files = {}
    for path in sorted(vector_dir.glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = vector_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"set_version": set_version, "files": files}, indent=1) + "\n"
    )
    return manifest_path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: gen_manifest.py <vector-dir>", file=sys.stderr)
        return 2
    path = write_manifest(argv[1], set_version=Path(argv[1]).name)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 4: Implement `tools/validate_vectors.py`**

Stdlib only. `validate_dir(path)` returns a list of violation strings, checking exactly the schema in this task's header: JSON parse of every `*.json`; required top-level fields and types; `verification` enum; `source` sha-or-reason plus nonempty `provenance`; `expected` has positive-int `kv_bytes_per_token` or `refuses: true`; optional ints positive when present; `experts` null or `{count, used}` ints; `windows` entries each have `terms` (obj), `usable_window` (int ≥ 0), `limited_by` (nonempty str); `must_not_equal` carries `note`; MANIFEST.json exists, lists exactly the vector files, and every sha matches the file bytes (recompute with hashlib). `main(argv)` mirrors gen_manifest: exit 0 / 1 (violations printed one per line) / 2 (usage).

- [ ] **Step 5: Run tests, verify they pass**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_validate_vectors.py -q`
Expected: 8 passed.

- [ ] **Step 6: Mutation check**

Purge `__pycache__` dirs. In `validate_vectors.py`, temporarily invert the sha comparison (`!=` → `==`); rerun; `test_manifest_sha_mismatch_is_a_violation` must FAIL. Restore. Then in `gen_manifest.py`, temporarily include `MANIFEST.json` in the glob loop; rerun; at least one test must FAIL (the manifest self-hash makes validation unstable). Restore, rerun, all green. If a mutation survives, the tests are defective — fix the tests before proceeding.

- [ ] **Step 7: Commit**

```bash
git add tools/ tests/ && git commit -m "feat: vector validator and manifest generator (stdlib, mutation-checked)"
```

---

### Task 3: Checker CLI

**Files:**
- Create: `tools/check.py`
- Test: `tests/test_check.py`

**Interfaces:**
- Produces: `check.compare(results: dict, vector_dir) -> Report` where `Report` is a frozen dataclass: `passed: list[str]`, `failed: list[tuple[str, str]]` (vector id, human-readable diff line), `uncovered: list[str]` (vector ids absent from results). CLI: `python3 tools/check.py <results.json> <vector-dir>` exits 0 (no failures) / 1 (any failure, incl. any `must_not_equal` hit) / 2 (usage/malformed). Prints per-vector PASS/FAIL/UNCOVERED lines and a summary. `uncovered` alone does NOT fail — consumers enforce coverage in their own tests.
- Consumes: vector schema from Task 2. Results file format (what a consumer emits): `{"<vector-id>": {"kv_bytes_per_token": int, "attention_layers": int|absent, "recurrent_state_bytes": int|absent, "serving_block_count": int|absent, "experts": null|{"count","used"}, "refuses": bool|absent, "windows": [{"usable_window": int, "limited_by": str}]|absent}}` — windows compared positionally against the vector's scenarios.

Comparison rules: every key present in the vector's `expected` must be present in the result and equal (ints exact; `experts: null` must be null — a `0` is a FAIL by R5). Keys in the result but not in `expected` are ignored (consumers may know more). Every `must_not_equal` field whose result value appears in the banned list is a FAIL naming the note.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_check.py
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
```

- [ ] **Step 2: Run, verify failure** — `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_check.py -q` → FAIL (`ModuleNotFoundError: check`).

- [ ] **Step 3: Implement `tools/check.py`** per the Interfaces block: load every vector (skip MANIFEST.json), walk `expected` keys, build diff strings `"<key>: expected <e>, got <g>"`, apply `must_not_equal` (`"<key>: got banned value <g> (<note>)"`), classify passed/failed/uncovered. `main(argv)`: read results JSON (exit 2 on parse/usage error), print lines, exit per contract. Frozen dataclass for `Report`.

- [ ] **Step 4: Run, verify pass** — 6 passed.

- [ ] **Step 5: Mutation check** — purge pycache; (a) make `must_not_equal` handling a no-op → `test_must_not_equal_hit_fails_naming_the_note` must fail; (b) treat vector-expected keys missing from the result as passing → `test_missing_expected_key_in_result_fails` must fail. Restore, all green.

- [ ] **Step 6: Commit** — `git add tools/check.py tests/test_check.py && git commit -m "feat: conformance checker CLI (compares, never derives)"`

---

### Task 4: Seed vectors — E1 attention corpus

**Files:**
- Create: `vectors/v1/<one JSON per model below>`
- Test: `tests/test_v1_vectors.py`

**Interfaces:**
- Consumes: validator (Task 2). Produces: the v1 vector files later tasks and consumers rely on; ids listed below are load-bearing.

Evidence sources (read-only): `~/workspace/assay/docs/superpowers/evidence/e1-sweep/` (`results.json` + `show-*.json` = verbatim `/api/show` responses; `model_info` IS the GGUF KV-pair dict — copy the geometry-relevant keys verbatim into the vector's `metadata`), plus the errata files beside committed profiles (locate: `grep -rln -i errata ~/workspace/assay/docs/`), plus `~/workspace/assay/docs/CHANGELOG.md` for corrected values.

Vectors to author (id = show-file stem). For EACH: metadata keys copied verbatim (`general.architecture`, `<arch>.block_count`, `<arch>.context_length`, `<arch>.embedding_length`, `<arch>.attention.head_count`, `<arch>.attention.head_count_kv`, `<arch>.attention.key_length` when present, expert keys when present); `expected.kv_bytes_per_token` copied from the corrected evidence (NOT recomputed by you — if evidence and your arithmetic disagree, STOP and flag it); `source.gguf_sha256` from the tags digest when it is a raw-GGUF sha, else `sha_unavailable_reason: "e1-sweep records daemon digest <digest>, not raw GGUF sha"`; `provenance` = the evidence path + errata path.

1. `qwen2.5-coder-7b-instruct-q8_0` — clean control. Anchor (verify against evidence, expect exactly): kv_bytes_per_token **57344** (56 KiB). Full example shape:

```json
{
  "id": "qwen2.5-coder-7b-instruct-q8_0",
  "source": {
    "sha_unavailable_reason": "e1-sweep records the daemon digest, not the raw GGUF sha",
    "provenance": "assay docs/superpowers/evidence/e1-sweep/show-qwen2.5-coder-7b-instruct-q8_0.json + results.json entry; kv 56 KiB/tok per assay live validation (docs/superpowers/evidence/2026-08-12-live-validation.md)"
  },
  "metadata": { "...": "copied verbatim from the show file's model_info" },
  "expected": {
    "kv_bytes_per_token": 57344,
    "experts": null,
    "windows": [
      {"terms": {"training_ctx": 32768, "budget_bytes": 2147483648, "weights_bytes": 0, "fixed_overhead_bytes": 536870912, "kv_bytes_per_token": 57344, "user_cap": null},
       "usable_window": 28086, "limited_by": "budget",
       "note": "synthetic terms; arithmetic: (2147483648-536870912)//57344 = 28086 < 32768"},
      {"terms": {"training_ctx": 32768, "budget_bytes": 8589934592, "weights_bytes": 0, "fixed_overhead_bytes": 536870912, "kv_bytes_per_token": 57344, "user_cap": null},
       "usable_window": 32768, "limited_by": "training_ctx",
       "note": "synthetic terms; kv fits 140434 tokens, training_ctx binds"},
      {"terms": {"training_ctx": 32768, "budget_bytes": 8589934592, "weights_bytes": 0, "fixed_overhead_bytes": 536870912, "kv_bytes_per_token": 57344, "user_cap": 8192},
       "usable_window": 8192, "limited_by": "user_cap",
       "note": "synthetic terms; user_cap binds"}
    ]
  },
  "verification": "hardware-verified"
}
```

These are the ONLY windows scenarios in v1 (they exercise R7's three binding terms with shown arithmetic; every other vector freezes interpretation values only — window numbers from live runs depended on unrecorded box VRAM and cannot be honestly frozen).

2. `qwen2.5-coder-14b-instruct-q4_K_M` — clean; expect kv **196608** (192 KiB, per tier-enthusiast evidence).
3. `qwen2.5-coder-1.5b-instruct-q8_0` — clean small control; value from evidence.
4. `codegemma-7b-instruct-q8_0` — AFFECTED. Corrected kv per errata (expect 448 KiB = **458752**; verify); `must_not_equal.kv_bytes_per_token` = the old-formula value (expect 336 KiB = 344064; copy from errata), note `"E1: embedding//head_count head_dim; corrected by R1 key_length rule"`.
5. `gemma2-9b` — AFFECTED, window HELD case. Corrected kv 336 KiB (verify vs errata); must_not_equal old 294 KiB value.
6. `mistral-nemo-latest` — AFFECTED, the under-promise sign. Corrected kv 160 KiB; must_not_equal old 200 KiB. Provenance notes the direction (window 32711→40889 under-promised).
7. ~~`qwen3.8-27b` — AFFECTED (16.8% window over-promise); corrected kv from errata; must_not_equal old value.~~ **WITHHELD — amended 2026-08-27 after Task 4 surfaced the conflict.** Its metadata states `full_attention_interval: 4`, `nextn_predict_layers: 1` and a full `ssm.*` set, so R3/R4/R6 apply; the errata's corrected 260 KiB (266240 B) charges all 65 raw blocks, a 4.06x attention over-charge against R3+R6, with R4's term uncharged. Pinning it would freeze the defect as the contract; deriving the conforming value here is banned. Filed as `docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md`; v1 ships **eight** E1 vectors, and R3/R4/R6 are covered by Task 5's hybrid instead.
8. `deepseek-coder-v2-16b-lite-instruct-q5_K_M` — AFFECTED + real MoE. Corrected kv from errata (33.3% window over-promise case); `experts: {"count": 64, "used": 6}`; must_not_equal old kv.
9. `gemma-4-12b-it-qat-q4_0-latest` — the R8 refusal case: e1-sweep classified it geometry:null (stated 512 vs derived 240, keys insufficient). `expected: {"refuses": true}`; provenance cites the results.json classification.

- [ ] **Step 1: Write the pinning test first**

```python
# tests/test_v1_vectors.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_vectors  # noqa: E402

V1 = Path(__file__).resolve().parents[1] / "vectors" / "v1"

EXPECTED_IDS = {
    "qwen2.5-coder-7b-instruct-q8_0", "qwen2.5-coder-14b-instruct-q4_K_M",
    "qwen2.5-coder-1.5b-instruct-q8_0", "codegemma-7b-instruct-q8_0",
    "gemma2-9b", "mistral-nemo-latest",
    "deepseek-coder-v2-16b-lite-instruct-q5_K_M",
    "gemma-4-12b-it-qat-q4_0-latest",
}  # amended 2026-08-27: qwen3.8-27b withheld, see item 7


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


def test_windows_only_on_the_designated_vector():
    for path in V1.glob("*.json"):
        if path.name == "MANIFEST.json":
            continue
        vec = json.loads(path.read_text())
        if vec["id"] != "qwen2.5-coder-7b-instruct-q8_0":
            assert "windows" not in vec["expected"], vec["id"]
```

- [ ] **Step 2: Run, verify failure** — vectors don't exist yet.

- [ ] **Step 3: Author the eight vectors** (nine as planned, minus the withheld item 7) per the recipe. For each: open the show file, copy the geometry-relevant `model_info` keys verbatim; copy corrected values from errata/CHANGELOG with the citation in `provenance`. **Discipline: if any anchor value in this plan disagrees with the evidence file, the evidence wins — stop, correct the plan value in the test, and record the discrepancy in the commit message.** Run `python3 tools/gen_manifest.py vectors/v1` after authoring.

- [ ] **Step 4: Run tests, verify pass** — `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/ -q` → all green.

- [ ] **Step 5: Cross-check with the checker** — build a tiny results file by hand from two vectors' expected values, run `python3 tools/check.py /tmp/spot.json vectors/v1`, confirm 2 PASS + 6 UNCOVERED (7 as planned, minus the withheld item 7), exit 0. Then corrupt one value, confirm FAIL + exit 1. (This is the checker's first run against real vectors — an end-to-end smoke, not a substitute for Task 3's tests.)

- [ ] **Step 6: Commit** — `git add vectors/ tests/test_v1_vectors.py && git commit -m "feat: v1 vectors — E1 attention corpus (8 models, provenance-cited)"`

---

### Task 5: Seed vectors — hybrid + mtp trap

**Files:**
- Create: `vectors/v1/qwen3.6-35b-a3b-reap48-ours-q4km.json`
- Create: `vectors/v1/qwen3.6-35b-a3b-reap48-mtp-trap.json`
- Modify: `tests/test_v1_vectors.py` (extend EXPECTED_IDS + anchors)

**Interfaces:**
- Consumes: validator, checker, Task 4's test module. Produces: the two hybrid vectors; ids above are load-bearing for consumer tasks.

- [ ] **Step 1: Extend the pinning test first** — add both ids to `EXPECTED_IDS`; add:

```python
def test_hybrid_anchor_values():
    v = load("qwen3.6-35b-a3b-reap48-ours-q4km")
    assert v["expected"]["kv_bytes_per_token"] == 20480
    assert v["expected"]["attention_layers"] == 10
    assert v["expected"]["recurrent_state_bytes"] == 65863680
    assert v["expected"]["serving_block_count"] == 40
    assert 81920 in v["must_not_equal"]["kv_bytes_per_token"]
    assert v["source"]["gguf_sha256"].startswith("90e2181e")

def test_mtp_trap_pins_serving_block_count():
    v = load("qwen3.6-35b-a3b-reap48-mtp-trap")
    assert v["metadata"].get("qwen3.5moe.mtp_num_hidden_layers", v["metadata"].get("mtp_num_hidden_layers")) == 1
    assert v["expected"]["serving_block_count"] == 40
    assert 41 in v["must_not_equal"]["serving_block_count"]
```

(If the real key name differs from `qwen3.5moe.mtp_num_hidden_layers`, correct the test to the name the dump shows — the *value* pins are the contract.)

Run; both fail (files absent).

- [ ] **Step 2: Extract the hybrid metadata**

```bash
sha256sum ~/models/gguf/qwen36-reap48-ours-Q4_K_M.gguf   # must start 90e2181e — STOP if not
python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py --no-tensors \
  ~/models/gguf/qwen36-reap48-ours-Q4_K_M.gguf > /tmp/claude-1000/-home-brice/*/scratchpad/reap48-dump.txt 2>/dev/null || \
python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py ~/models/gguf/qwen36-reap48-ours-Q4_K_M.gguf | head -100
```

(External authoring tool; nothing from llama.cpp is shipped. If `--no-tensors` is not a valid flag, use the plain form and take the KV section.) Copy the geometry-relevant keys verbatim: architecture, block_count, `full_attention_interval`, attention head counts / key_length, all `ssm.*` keys, expert_count, expert_used_count, context_length.

- [ ] **Step 3: Author the hybrid vector** — expected values with provenance `"bloomery docs/superpowers/evidence/2026-08-21-reap48-qwen36-spike.md + 2026-08-22-g5v4-reap48-baselines.md (two boots, hardware-verified)"`, verification `hardware-verified`, `must_not_equal`: kv 81920 (note: `"all-40-blocks derivation — R3 violation, fixed bloomery turn 5"`). Verify each expected number against those two bloomery docs (read-only) before freezing; evidence wins over this plan.

- [ ] **Step 4: Author the mtp-trap vector** — metadata = the hybrid vector's keys PLUS the mtp key at 1 and block_count 41 (the pre-patch trap state, reconstructed); expected `serving_block_count: 40`, kv/attention/recurrent as the hybrid vector; `must_not_equal.serving_block_count: [41]` (note: `"R6: MTP layer counted as a serving layer → unloadable GGUF, bloomery CARRIED-DEBT 2026-08-22"`); `sha_unavailable_reason: "trap state reconstructed from evidence; the patched artifact no longer carries it"`, verification `derived`.

- [ ] **Step 5: Regenerate manifest, run everything**

```bash
python3 tools/gen_manifest.py vectors/v1
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/ -q
python3 tools/validate_vectors.py vectors/v1
```
Expected: all green, exit 0.

- [ ] **Step 6: Commit** — `git commit -am "feat: v1 vectors — REAP-48 hybrid + mtp trap (10 vectors total)"`

---

### Task 6: Repo CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: ci
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: python -m pip install pytest
      - run: python -m pytest tests/ -q
      - run: python tools/validate_vectors.py vectors/v1
```

- [ ] **Step 2: Local dry-run of exactly what CI runs** — both commands green, validator exit 0.

- [ ] **Step 3: Commit** — `git add .github && git commit -m "ci: pytest + vector validation on push"`

---

### Task 7: HUMAN GATE — publish

**Files:** none (operations only).

- [ ] **Step 1: Secret scan** — `grep -rniE 'api[_-]?key|token|secret|password|BEGIN [A-Z]+ PRIVATE' --include='*' . | grep -v '^\.git/'` from the repo root; also confirm no absolute paths leak beyond `~/workspace`-style doc references. Expected: no hits outside documentation prose.
- [ ] **Step 2: STOP — present to Brice**: repo tree, test count, vector count. Wait for explicit go.
- [ ] **Step 3 (after go):** `gh repo create bricelancasterwcp-sudo/gguf-geometry --public --source . --push` (https remote — SSH fails for new repos on this box, see github-sync-after-push). Then verify sync: `git fetch origin && git status -sb` shows `## master...origin/master` with no ahead/behind, and `gh repo view --json defaultBranchRef` resolves. CI green on GitHub.

---

### Task 8: assay conformance harness (branch, MERGE HOLD)

**Files:**
- Create (in assay worktree): `tests/data/gguf_geometry_v1/` (vendored copy of `vectors/v1/`)
- Test: `tests/test_geometry_conformance.py`

**Interfaces:**
- Consumes: v1 vectors; assay's own `ModelInfo` construction and `assay.geometry.kv_bytes_per_token` / `plan_window`.
- Produces: a red/green conformance report consumed by Task 9. Branch `geometry-conformance`, worktree `~/workspace/assay/.worktrees/geometry-conformance`.

**MERGE HOLD (spec §6): this branch is never merged, pushed, or rebased onto master until Brice declares the bloomery memory-battery closed.** bloomery's deployed assay follows a PYTHONPATH pin; a mid-battery master change risks altering the battery's probe layer in place.

- [ ] **Step 1: Create the worktree**

```bash
cd ~/workspace/assay && git worktree add .worktrees/geometry-conformance -b geometry-conformance
```

**venv gotcha (recorded 2026-08-19): assay's .venv imports the MAIN checkout's src. Every test run in this worktree MUST use `PYTHONPATH=src` from the worktree root**, e.g. `cd .worktrees/geometry-conformance && PYTHONPATH=src python3 -m pytest ...` — verify with `python3 -c "import assay, sys; print(assay.__file__)"` that the worktree path wins.

- [ ] **Step 2: Vendor the vectors** — copy `~/workspace/gguf-geometry/vectors/v1/*` to `tests/data/gguf_geometry_v1/`; record the vendored MANIFEST's own sha256 as a constant in the test module (the pin: re-vendoring is a deliberate, visible diff).

- [ ] **Step 3: Find assay's interpretation seam** — `grep -n 'model_info\|ModelInfo(' src/assay/backends/ollama.py`. The function that builds `ModelInfo` from an `/api/show` payload is the code under test — the conformance test feeds each vector's `metadata` through IT (wrapped as `{"model_info": metadata, ...}` in whatever payload shape it expects), never through a hand-mapping in the test. Read the call sites to get the exact payload shape and function name before writing the test.

- [ ] **Step 4: Write the conformance test**

```python
# tests/test_geometry_conformance.py  (names per Step 3 findings)
import hashlib
import json
from pathlib import Path

import pytest

from assay.geometry import kv_bytes_per_token
# + the ModelInfo-builder import found in Step 3

DATA = Path(__file__).parent / "data" / "gguf_geometry_v1"
VENDORED_MANIFEST_SHA = "<sha256 of the vendored MANIFEST.json — fill at vendoring>"


def vectors():
    for path in sorted(DATA.glob("*.json")):
        if path.name != "MANIFEST.json":
            yield json.loads(path.read_text())


def test_vendored_manifest_is_pinned():
    actual = hashlib.sha256((DATA / "MANIFEST.json").read_bytes()).hexdigest()
    assert actual == VENDORED_MANIFEST_SHA


@pytest.mark.parametrize("vec", list(vectors()), ids=lambda v: v["id"])
def test_kv_interpretation_conforms(vec):
    info = build_model_info_from_show({"model_info": vec["metadata"]})  # Step-3 name
    kv = kv_bytes_per_token(info)
    if vec["expected"].get("refuses"):
        assert kv is None, "R8: unknown geometry must refuse, not guess"
        return
    banned = vec.get("must_not_equal", {}).get("kv_bytes_per_token", [])
    assert kv not in banned, f"banned historical value: {vec['must_not_equal'].get('note')}"
    assert kv == vec["expected"]["kv_bytes_per_token"]
```

Plus one `plan_window` test driving the three windows scenarios of the qwen-7b vector through `plan_window` (map `budget_bytes` → `vram_free_mib = budget_bytes // 2**20 + overhead_mib` with `loaded=True`, `user_cap` → `user_cap`; assert `usable_window` and the mapped `limited_by` — assay's `"vram"` corresponds to the vector's `"budget"`).

- [ ] **Step 5: Run and RECORD the reds honestly**

```bash
cd ~/workspace/assay/.worktrees/geometry-conformance && PYTHONPATH=src python3 -m pytest tests/test_geometry_conformance.py -v 2>&1 | tail -30
```
Predicted: E1-corpus vectors PASS (key_length rule shipped in v1.6); `qwen3.6-35b-a3b-reap48-ours-q4km` and the mtp trap FAIL (no R3/R4/R6 handling). Write the actual red list into the commit message. **Do not weaken assertions, xfail, or skip to get green — the reds are Task 9's work order.** If an *E1-corpus* vector unexpectedly fails, STOP: either the vector is mis-authored (fix in gguf-geometry, re-vendor — new manifest sha) or assay has an unknown bug (report to Brice before proceeding).

- [ ] **Step 6: Commit on the branch** — `git add tests/ && git commit -m "test: gguf-geometry v1 conformance harness (vendored, sha-pinned; hybrid vectors red as expected)"`

---

### Task 9: assay hybrid geometry fix (same branch, MERGE HOLD)

**Files:**
- Modify (worktree): `src/assay/geometry.py`, `src/assay/backends/base.py` (ModelInfo fields), `src/assay/backends/ollama.py` (+ `openai_compat.py` if it builds ModelInfo too — check), `CHANGELOG.md`
- Test: extend `tests/test_geometry_conformance.py` expectations only if needed; add unit tests beside assay's existing geometry tests (locate: `grep -rln plan_window tests/ src/`)

**Interfaces:**
- Consumes: Task 8's red list. Produces: green conformance suite; new `ModelInfo` fields `attention_layer_count: int | None = None`, `recurrent_state_bytes: int | None = None`, `mtp_layer_count: int | None = None` — **all None-defaulted so every existing profile and caller parses unchanged** (the v1.6 expert-fields precedent: additive, no schema break).

Scope rule from the spec: fix the pure derivation given the keys. Plumbing the new keys through *live* backend probing beyond what the show-payload parser already sees is assay-wave work — if it turns out to be required for anything beyond the parser function, record it as CARRIED-DEBT instead of expanding this task.

- [ ] **Step 1: Write failing unit tests first** (beside existing geometry tests): `kv_bytes_per_token` honors `attention_layer_count` when present (R3) and falls back to `block_count` when absent (dense unchanged — regression-pin an existing dense case); serving block count subtracts `mtp_layer_count` (R6); recurrent bytes pass through to `Geometry` (R4) as a reported field, charged in `plan_window`'s vram term as a fixed per-context subtraction. Run; red.
- [ ] **Step 2: Implement** — parser: read `full_attention_interval`, `ssm.*`, `mtp_num_hidden_layers` keys (exact names from the vendored REAP-48 vector's metadata; bloomery's `crates/bloomery-core/src/gguf.rs` is the read-only reference for ssm-dim → bytes arithmetic — copy the semantics, cite the file in a comment only if assay convention does so — check neighboring comments first). geometry: `attention_layer_count = serving_block_count // full_attention_interval` when the key exists; `Geometry` gains the None-defaulted reporting fields.
- [ ] **Step 3: Full suite green** — `PYTHONPATH=src python3 -m pytest tests/ -q` (worktree): conformance 11/11 + entire existing suite unchanged. Any existing-test change must be justified line-by-line in the commit message.
- [ ] **Step 4: Mutation check** — break the R3 division (use raw block_count); conformance hybrid vector AND the new unit test must both fail (the `must_not_equal: 81920` pin is the designed catch). Restore.
- [ ] **Step 5: CHANGELOG entry** under an Unreleased heading (follow the file's existing format): hybrid geometry (R3/R4/R6) added, fields None-defaulted, no schema bump — committed profiles unaffected because no committed profile is a hybrid model (verify: `grep -rl full_attention_interval docs/` returns nothing under committed profiles; if it does, STOP — errata territory, Brice decides).
- [ ] **Step 6: Commit on the branch; report** — branch stays local. Report to Brice: red→green list, the held-merge state, and that the branch unblocks when the battery closes.

---

### Task 10: HUMAN GATE — bloomery conformance (post-battery only)

**Precondition: Brice has declared the memory-battery arc CLOSED and its worktree merged/cleaned. Do not start — do not even run `cargo` in bloomery — before that.**

**Files:**
- Create: `crates/bloomery-core/tests/data/gguf_geometry_v1/` (vendored copy)
- Test: `crates/bloomery-core/tests/geometry_conformance_test.rs`

**Interfaces:**
- Consumes: v1 vectors; `bloomery_core::gguf::GgufMeta`, `bloomery_core::geometry::{kv_bytes_per_token, usable_window, GeometryInput, Window, BoundBy}` (signatures verified 2026-08-27: `kv_bytes_per_token(&GgufMeta) -> u64`, `usable_window(&GeometryInput) -> Window`).

Honest scope note (spec §6): bloomery's interpretation layer lives inside `parse_gguf_meta` (binary reader), which JSON vectors cannot drive. This task tests the *derivation* functions by constructing `GgufMeta` from vector metadata values in the test; interpretation coverage remains bloomery's existing real-GGUF tests (`gguf_real_hybrid_test.rs`). State this in the test module's doc comment.

- [ ] **Step 1: Vendor vectors + write the test** — build `GgufMeta` per vector from its metadata values (field mapping read from `gguf.rs:16`'s struct — fill only fields the derivation consumes; if a needed field is private or unconstructable, add a `#[cfg(test)]` constructor to gguf.rs rather than making fields pub). Assert `kv_bytes_per_token` == expected and != every banned value; hybrid vector asserts attention-layer and recurrent terms via the struct's fields; drive the three qwen-7b windows scenarios through `usable_window` mapping terms onto `GeometryInput`.
- [ ] **Step 2: Run** — `cargo test -p bloomery-core --test geometry_conformance_test`. Predicted green (bloomery already implements R3/R4/R6); any red is a real finding — report, don't paper over.
- [ ] **Step 3: BOX RULE — rebuild the featured binary**: `cargo build -p bloomery-daemon --features vulkan` (cargo test clobbers it featureless; a later boot would fail to load models).
- [ ] **Step 4: Commit on a branch (`geometry-conformance`), full workspace suite green, PR per bloomery convention; Brice merges.**

---

## Self-Review (performed at write time)

- **Spec coverage:** §0/§1 → Task 1 + global constraints; §2 → Task 1 (R1–R8); §3 → Tasks 2, 4, 5; §4 → Tasks 4, 5; §5 → Task 3; §6 sequencing + merge hold → Tasks 7–10 gates; §7 acceptance → Tasks 4–6 (repo CI), 8–10 (consumers; the "divergence surfaced and resolved" clause = Task 8 reds → Task 9 green); §8 → Task 7.
- **Known open value:** `VENDORED_MANIFEST_SHA` is filled at vendoring time by design (it cannot exist before the vectors are frozen) — not a placeholder.
- **Type consistency:** vector field names (`kv_bytes_per_token`, `serving_block_count`, `attention_layers`, `recurrent_state_bytes`, `experts{count,used}`, `refuses`, `windows{terms,usable_window,limited_by}`) are identical across Tasks 2, 3, 4, 5, 8, 10. Checker `Report` fields (`passed/failed/uncovered`) consistent between Task 3 code and prose.
- **Anchor-value honesty:** plan anchors (57344, 196608, 20480, 10, 65863680, 40/41, 81920, experts 64/6, sha prefix 90e2181e) are from committed evidence read this session or memory verified against it; every authoring step re-verifies against the evidence file and the evidence wins on conflict.
