# MLA KV Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what ollama 0.32.13 actually allocates per KV-cache token for deepseek2 (MLA), then close the SPEC's "Known open gap: R2 and MLA" with the rule the measurement supports, cut `vectors/v3`, and bring both consumers (assay, bloomery) onto it.

**Architecture:** Three phases across three repos, gated on Brice's rulings. Phase 1 (assay) commits a pre-registration protocol BEFORE any capture, then captures raw evidence, then applies the pre-registered decision rule. Phase 2 (gguf-geometry) writes the outcome-mapped rule + v3 vector set. Phase 3 implements the rule in assay's and bloomery's geometry seams and re-vendors v3.

**Tech Stack:** bash/journalctl/nvidia-smi (capture), Python + pytest via uv (gguf-geometry, assay), Rust/cargo (bloomery).

**Spec:** `docs/superpowers/specs/2026-08-27-mla-kv-rule-design.md` (this repo). The spec and the committed PROTOCOL.md are the binding authority; where this plan and they disagree, **they win** (2026-08-27 lesson: plan text loses to spec, twice).

## Global Constraints

- **Pre-registration is inviolable**: no journalctl inspection, no model load, no peek at any KV number before Task 1's commit exists. The point estimate decides; no re-run, no candidate added, no band widened after a number is seen. Infrastructure failure (failed load, OOM, missing log line) is recorded as infrastructure, never as data, and never re-rolls a completed reading.
- **Phase gates**: STOP after Task 3 (verdict → Brice ruling), after Task 5 (gguf-geometry merge/push ruling), after Task 8 (assay ruling), after Task 10 (bloomery ruling). Merges and pushes only on explicit rulings.
- **Branch discipline**: Tasks 4–10 are written for all four pre-registered outcomes (H-a, H-b, H-c, H-c′) plus no-match. Execute ONLY the branch VERDICT.md names. No-match → Tasks 4–10 do not run (SPEC gap paragraph gains the measured figure instead; see Task 4 no-match note).
- Test runners (handoff-pinned): assay & gguf-geometry: `PYTHONDONTWRITEBYTECODE=1 uv run --offline --with pytest --python 3.12 python -m pytest -q` (bare python3 is 3.14, no pytest); bloomery: `cargo test --workspace`, then ALWAYS `cargo build -p bloomery-daemon --features vulkan` LAST (clobber rule).
- Every verification run: full output to a file under the repo's scratch or `/tmp/claude-*` + `echo exit=$?`; never pipe through tail/head without capturing exit.
- Mutation checks: revert then `touch` the file (never timestamp-preserving restore); purge `__pycache__`; `PYTHONDONTWRITEBYTECODE=1`.
- Commit style: conventional commits, no attribution trailers (disabled globally).
- Model of record: `deepseek-coder-v2:16b-lite-instruct-q5_K_M`, GGUF sha256 `bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406`. Runtime of record: ollama 0.32.13, systemd unit `ollama`, box `brice-X870E-Taichi`, RTX 5080 16 GiB.
- Candidate table (arithmetic fixed by the spec; attention_layers 27, head_count_kv 16, key_length 192, value_length 128, kv_lora_rank 512, rope.dimension_count 64, f16 = 2 B/elt):

| Id | Formula | B/token | ±5% band |
|----|---------|---------|----------|
| H-a | 2·27·16·192·2 | 331,776 | 315,188 – 348,364 |
| H-b | 27·16·(192+128)·2 | 276,480 | 262,656 – 290,304 |
| H-c | 27·(512+64)·2 | 31,104 | 29,549 – 32,659 |
| H-c′ | 2·27·(512+64)·2 | 62,208 | 59,098 – 65,318 |

---

## Phase 1: Measurement (repo: `~/workspace/assay`)

### Task 1: Pre-registration protocol, committed before anything runs

**Files:**
- Create: `docs/superpowers/evidence/mla-kv-2026-08-27/PROTOCOL.md` (if execution crosses midnight into 08-28, date the dir by capture day and use that name consistently everywhere below)

**Interfaces:**
- Produces: the committed protocol every later task obeys; its dir path (call it `EVIDENCE_DIR`) is used by Tasks 2–3, and its verdict vocabulary (`H-a|H-b|H-c|H-c'|no-match`) by Tasks 4–10.

- [ ] **Step 1: Sanity-check the candidate arithmetic** (a chosen band rides on it):

```bash
python3 -c "
a = 2*27*16*192*2; b = 27*16*(192+128)*2; c = 27*(512+64)*2; c2 = 2*c
assert (a,b,c,c2) == (331776,276480,31104,62208), (a,b,c,c2)
for lo,hi in [(int(x*0.95),int(x*1.05)) for x in (a,b,c,c2)]: print(lo,hi)
print('closest pair gap:', a/b)
"
```
Expected: the four bands printed match the table above; closest pair gap 1.2 (20%, so the bands cannot overlap).

- [ ] **Step 2: Write `PROTOCOL.md`** with exactly these sections (prose may be tightened, commitments may not):

```markdown
# Pre-registered protocol: deepseek2 KV bytes/token under ollama 0.32.13

Written and committed BEFORE any capture. Subject: what this runtime
actually allocates per KV-cache token for
deepseek-coder-v2:16b-lite-instruct-q5_K_M (GGUF sha256 bc286970…2406).
Lens: ollama 0.32.13 (systemd unit), this box (RTX 5080 16 GiB), KV
cache type f16 (OLLAMA_KV_CACHE_TYPE unset), point-in-time. This is a
property of the runtime-on-this-box, not of the GGUF.

## Candidates
[the four-row table from the plan, verbatim, with the formulas]
H-c' is llama.cpp's historical MLA variant caching a transposed copy of
the latent; a derived implementation variant, not a post-hoc hedge.

## Decision rule
Instrument 1's bytes/token matching exactly ONE candidate within ±5%
decides. Bands are disjoint (closest pair 20% apart). No match → no
rule is written; the figure is recorded and the SPEC gap stays open.
The point estimate decides: no re-run, no extension, no added candidate
after a number is seen. A failed load / missing log line / OOM is
infrastructure, recorded as such, never data.

## Instrument 1 (primary): server-log KV buffer lines
Load with explicit options.num_ctx; slice journalctl for the load;
quote VERBATIM every KV-cache buffer/size line (all backends; if the
cache splits across CUDA0/CPU, the reading is the SUM of the lines).
bytes/token = summed KV bytes ÷ the n_ctx THE LOG REPORTS (not the
requested one; parallel slots inflate effective n_ctx). Every "because"
in the reading is a quoted line.

## Instrument 2 (corroboration only): VRAM slope
num_ctx ∈ {2048, 4096, 8192}; nvidia-smi before/after each load.
Validity per point: ollama ps reports 100% GPU, else the point is named
in dropped. Stated confound: compute buffers also grow with ctx, so the
slope over-reads KV. Pre-registered consequence: instrument 2 CANNOT
decide alone; it must only be consistent with instrument 1 (same order
of magnitude, slope ≥ instrument 1's figure).

## Contingency
If ollama 0.32.13's logs surface no usable KV-buffer lines, fallback is
the same blob under a standalone llama-server, a DIFFERENT NAMED LENS;
whether a rule may be written from it is Brice's ruling, not a swap.

## Ambient record (filled at capture, committed with the evidence)
ollama version; systemd env (OLLAMA_KV_CACHE_TYPE, OLLAMA_FLASH_ATTENTION,
OLLAMA_NUM_PARALLEL); runner engine as logged; GPU + free VRAM before
each point; ollama ps residency per point; timestamps.
```

- [ ] **Step 3: Commit** (protocol alone, before any capture):

```bash
cd ~/workspace/assay && git add docs/superpowers/evidence/mla-kv-2026-08-27/PROTOCOL.md \
  && git commit -m "docs: pre-register MLA kv measurement protocol (deepseek2 under ollama 0.32.13)"
```

### Task 2: Capture raw evidence (no interpretation in this task)

**Files:**
- Create: `EVIDENCE_DIR/{preflight.txt, env.txt, raw-log-ctx2048.txt, raw-log-ctx4096.txt, raw-log-ctx8192.txt, vram-ctx*.txt, ps-ctx*.txt}`

**Interfaces:**
- Consumes: committed PROTOCOL.md (Task 1).
- Produces: raw capture files Task 3 reads. Nothing else may read numbers out of them before Task 3.

- [ ] **Step 1: Preflight, digest assert + GPU hygiene** (store of record, not guessed paths):

```bash
cd ~/workspace/assay && EV=docs/superpowers/evidence/mla-kv-2026-08-27
M=$(ls /mnt/extra/ollama-models/manifests/registry.ollama.ai/library/deepseek-coder-v2/*16b-lite-instruct-q5_K_M* 2>/dev/null | head -1)
{ echo "manifest: $M"; python3 -c "
import json,sys; m=json.load(open('$M'))
layer=[l for l in m['layers'] if l['mediaType'].endswith('model')][0]
d=layer['digest']; print('model layer digest:', d)
assert d=='sha256:bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406', 'DIGEST MISMATCH: STOP'
print('digest OK')"
  ollama --version; ollama ps; nvidia-smi --query-gpu=memory.used,memory.free --format=csv; } | tee $EV/preflight.txt
```
Expected: `digest OK`; `ollama ps` empty (unload anything listed with `ollama stop <name>` and re-run). If the manifest path glob misses, find it with `grep -rl 16b-lite /mnt/extra/ollama-models/manifests`. Never proceed without the digest assert.

- [ ] **Step 2: Record ambient env**:

```bash
cd ~/workspace/assay && EV=docs/superpowers/evidence/mla-kv-2026-08-27
{ systemctl cat ollama | grep -i env; systemctl show ollama -p Environment; date -u -Is; } | tee $EV/env.txt
```
Expected: `OLLAMA_KV_CACHE_TYPE` absent/unset (f16 default). If it is set to anything else: STOP. The protocol pins f16; record and ask Brice.

- [ ] **Step 3: Capture loop, one load per ctx point** (2048, 4096, 8192):

```bash
cd ~/workspace/assay && EV=docs/superpowers/evidence/mla-kv-2026-08-27
MODEL=deepseek-coder-v2:16b-lite-instruct-q5_K_M
for CTX in 2048 4096 8192; do
  T0=$(date -u '+%Y-%m-%d %H:%M:%S')
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader > $EV/vram-ctx$CTX.txt
  curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"options\":{\"num_ctx\":$CTX},\"keep_alive\":\"3m\"}" > /dev/null
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader >> $EV/vram-ctx$CTX.txt
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> $EV/vram-ctx$CTX.txt
  ollama ps > $EV/ps-ctx$CTX.txt
  journalctl -u ollama --since "$T0" --no-pager > $EV/raw-log-ctx$CTX.txt
  ollama stop $MODEL; sleep 5
done; echo exit=$?
```
Expected: three raw-log files each containing the runner's load lines. Do NOT read KV figures out of them yet. Task 3 does the reading against the committed protocol.

- [ ] **Step 4: Commit the raw captures**:

```bash
cd ~/workspace/assay && EV=docs/superpowers/evidence/mla-kv-2026-08-27
git add $EV && git commit -m "docs: raw capture of the deepseek2 kv measurement, 3 ctx points (protocol-first)"
```

### Task 3: Reading + verdict, then STOP for ruling

**Files:**
- Create: `EVIDENCE_DIR/READING.md`, `EVIDENCE_DIR/VERDICT.md`

**Interfaces:**
- Consumes: Task 2's raw files, Task 1's decision rule.
- Produces: `VERDICT.md` naming exactly one of `H-a|H-b|H-c|H-c'|no-match`, the branch selector every later task obeys.

- [ ] **Step 1: Extract instrument-1 lines verbatim.** From each `raw-log-ctx*.txt`, quote into READING.md: the runner identification line (llama runner vs ollama engine), every line matching `grep -iE "kv|cache" raw-log-ctx*.txt` that states a buffer size or n_ctx, and the log-reported n_ctx. Compute bytes/token per point = summed KV buffer bytes ÷ log-reported n_ctx, arithmetic shown.
- [ ] **Step 2: Instrument-2 slope.** From the `vram-ctx*.txt` before/after pairs: per-point deltas and the slope across valid points; mark any point whose `ps-ctx*.txt` is not `100% GPU` as `dropped: [point, reason]`. State consistency-or-not with instrument 1 (order of magnitude, slope ≥).
- [ ] **Step 3: Apply the decision rule.** One candidate within ±5% on instrument 1 (all points must agree on the candidate) → that candidate; else `no-match`. Write VERDICT.md: the verdict, the instrument-1 figure(s), the band arithmetic, instrument-2 consistency line, and the ambient lens summary (runtime, box, cache type, engine, date).
- [ ] **Step 4: Commit and STOP**:

```bash
cd ~/workspace/assay && EV=docs/superpowers/evidence/mla-kv-2026-08-27
git add $EV && git commit -m "docs: MLA kv measurement reading + verdict (pre-registered decision rule applied)"
```
Report the verdict to Brice with the quoted lines. **Do not begin Phase 2 without a ruling.** If the contingency fired (no usable lines), present the llama-server lens question instead. That needs its own ruling before ANY further capture.

---

## Phase 2: Rule + vectors/v3 (repo: `~/workspace/gguf-geometry`)

### Task 4: SPEC amendment (+ erratum doc if H-a lost)

**Files:**
- Modify: `SPEC.md` (R2 block lines 14–21; new R9 after R8 if H-b/H-c/H-c′; Versioning section, adding a new v3 paragraph; "Known open gap" paragraph lines 93–105 amended non-silently)
- Create (only if verdict ≠ H-a): `docs/upstream-errata/2026-08-27-assay-deepseek-mla-kv-overcharge.md`

**Interfaces:**
- Consumes: `VERDICT.md` (Task 3).
- Produces: rule text that Task 5's vector provenance cites and Tasks 6–10 implement. Rule name: `R9` (H-b/c/c′ branches) or amended `R2` (H-a).

- [ ] **Step 1: Amend SPEC.md per branch.** All branches: append to the "Known open gap" paragraph (never rewrite it): `**Closed YYYY-MM-DD by measurement.** assay docs/superpowers/evidence/mla-kv-2026-08-27/ (ollama 0.32.13, this box): <verdict + figure>. See <R2 amendment | R9> and vectors/v3.` Branch texts:
  - **H-a**: amend R2 in place, dated: `(Amended 2026-08-27: measured, ollama 0.32.13 allocates the key width for both K and V on deepseek2; a stated attention.value_length ≠ key_length does not change the cache size under the measured runtime. Origin: assay mla-kv-2026-08-27.)`
  - **H-b**: add R9: `## R9: MLA, separate widths\nWhen <arch>.attention.value_length is present and differs from attention.key_length: kv_bytes_per_token = attention_layer_count * head_count_kv * (key_length + value_length) * bytes_per_element. R2 defers to this rule in that case. (Origin: measured on deepseek2 under ollama 0.32.13, assay mla-kv-2026-08-27; one architecture, one runtime, dated.)`
  - **H-c / H-c′**: add R9: `## R9: MLA, latent cache\nWhen <arch>.attention.value_length is present and differs from attention.key_length: kv_bytes_per_token = [2 * (H-c′ only)]attention_layer_count * (attention.kv_lora_rank + rope.dimension_count) * bytes_per_element, per layer, no head factor. If kv_lora_rank or rope.dimension_count is absent in that case, refuse (R8). (Origin: measured on deepseek2 under ollama 0.32.13, assay mla-kv-2026-08-27.)`
  - **All decisive branches**: Versioning gains: `**v3 (YYYY-MM-DD).** vectors/v3/: eleven vectors, ten carried byte-for-byte from v2, deepseek-coder-v2 re-pinned to the measured rule (<R9|R2-confirmed>), disproved candidates pinned as must_not_equal. v1/v2 stay frozen.`
  - **no-match**: ONLY the open-gap paragraph gains the measured figure + lens; no rule, no v3; skip Tasks 5–10 and report.
- [ ] **Step 2 (verdict ≠ H-a only): write the erratum doc.** E3-style, mirroring `docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md` in structure: what was published (deepseek kv 324 KiB/token, "hardware-verified"), what it actually was (R2 arithmetic executed on a live box (`geometry.source: api_show`), never an observed allocation), the measured figure + lens, direction (over-charge → windows under-promised), affected artifacts (assay tier profiles, gguf-geometry v1/v2 deepseek vector `expected`. Frozen sets stay frozen; the erratum is the record), and the fix chain (R9/v3, assay E3, both consumers).
- [ ] **Step 3: Run the repo suite** (SPEC-referencing tests may pin wording):

```bash
cd ~/workspace/gguf-geometry && PYTHONDONTWRITEBYTECODE=1 uv run --offline --with pytest --python 3.12 python -m pytest -q > /tmp/claude-gg-task4.txt 2>&1; echo exit=$?
```
Expected: exit=0 (v1/v2 tests untouched by SPEC prose; if a test pins the open-gap wording, amend the TEST's expectation only where the spec says the text gained an appended closure. Never weaken a frozen-set pin).
- [ ] **Step 4: Commit** `docs: close R2/MLA gap per measurement, <branch> (+ E3 erratum)`.

### Task 5: vectors/v3 + test_v3_vectors.py + MANIFEST (TDD), then STOP for ruling

**Files:**
- Create: `vectors/v3/` (11 vector files + `MANIFEST.json`), `tests/test_v3_vectors.py`
- Modify: `vectors/v3/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json` (the one vector that changes)

**Interfaces:**
- Consumes: Task 4's rule name; Task 3's verdict; `tools/gen_manifest.py`, `tools/validate_vectors.py` (unchanged).
- Produces: the v3 set consumers vendor in Tasks 8/10; `vectors/v3/MANIFEST.json` shas they assert.

- [ ] **Step 1: Write `tests/test_v3_vectors.py` first.** Copy `tests/test_v2_vectors.py` as the base and change: `V2 = ROOT/"vectors"/"v2"` → carried-set constant `V3 = ROOT/"vectors"/"v3"`; `UPGRADED` → exactly one entry, the deepseek file, whose allowed diffs are: the branch's metadata additions (below), the new `expected.kv_bytes_per_token`, the extended `must_not_equal` list, and the provenance sentence; a new pin asserting the deepseek vector's expected equals the branch value and its must_not_equal contains every disproved candidate; carried-forward pin: the OTHER TEN files byte-identical to v2 (`cmp`-equality via file bytes, the v2 test's mechanism). Branch parameter table (single source for this task):

| Verdict | expected.kv_bytes_per_token | must_not_equal adds (kept: 221184) | metadata adds |
|---|---|---|---|
| H-a | 331776 (unchanged) | 276480, 31104, 62208 | none |
| H-b | 276480 | 331776, 31104, 62208 | none |
| H-c | 31104 | 331776, 276480, 62208 | `deepseek2.attention.kv_lora_rank: 512`, `deepseek2.rope.dimension_count: 64` |
| H-c′ | 62208 | 331776, 276480, 31104 | same as H-c |

- [ ] **Step 2: Run it. It must FAIL** (`vectors/v3` doesn't exist):

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --with pytest --python 3.12 python -m pytest -q tests/test_v3_vectors.py > /tmp/claude-gg-task5-red.txt 2>&1; echo exit=$?
```
Expected: nonzero exit, `vectors/v3` missing.
- [ ] **Step 3: Build the set**: `mkdir vectors/v3 && cp -a vectors/v2/*.json vectors/v3/ && rm vectors/v3/MANIFEST.json` (carry all, then edit the one). Edit the deepseek vector: apply the branch row from the table (must_not_equal each gains a note clause naming what it was: `"E1: embedding//head_count head_dim"` stays; append to note: `"; MLA candidates disproved by measurement (assay mla-kv-2026-08-27): <list>. See SPEC <R9|R2 amendment>."`). Append to `source.provenance` (never rewrite): `v3 (YYYY-MM-DD): expected re-pinned per <R9|confirmed R2>, measured <figure> B/token under ollama 0.32.13 (assay docs/superpowers/evidence/mla-kv-2026-08-27/, instrument 1 log lines); disproved candidates pinned. [H-c/H-c′ only:] kv_lora_rank 512 and rope.dimension_count 64 stated verbatim from the same show capture the vector already cites.` `verification` stays `"hardware-verified"`, and for the first time the kv figure is an OBSERVED allocation, which the provenance sentence says.
- [ ] **Step 4: Regenerate manifest**: `uv run --offline --python 3.12 python tools/gen_manifest.py vectors/v3` (check `tools/gen_manifest.py --help`/source first, and invoke exactly as the v2 arc did; the v2 MANIFEST.json is the shape reference).
- [ ] **Step 5: Full suite green** (same runner command as Task 4, output to `/tmp/claude-gg-task5-green.txt`; both validators exit 0: `uv run --offline --python 3.12 python tools/validate_vectors.py vectors/v3` and the v2 invocation the CI workflow runs).
- [ ] **Step 6: Mutation-check the new pins** (each: mutate → suite FAILS → revert + `touch`): (1) flip the deepseek expected to a disproved candidate → carried/expected pin must fail; (2) delete one disproved candidate from must_not_equal → the new pin must fail; (3) alter one byte of a carried-forward vector → byte-identity pin must fail. Record which assert fired, quoted.
- [ ] **Step 7: Commit** `feat: v3 vector set, MLA rule pinned by measurement (<branch>)`. **STOP.** Present the Task 4+5 diff to Brice for the gguf-geometry merge/push ruling (CI must be green on push before Phase 3 vendors anything).

---

## Phase 3a: assay implements + re-vendors (repo: `~/workspace/assay`)

**H-a note:** Tasks 6 and 7 are SKIPPED under H-a (no behavior change). Task 8 still runs (v3 re-vendor; no E3, no SEMANTIC_BREAKS row, because the semantic did not move).

### Task 6: Backend reads the MLA keys into ModelInfo

**Files:**
- Modify: `src/assay/backends/base.py` (ModelInfo, after `mtp_layer_count` ~line 243), `src/assay/backends/ollama.py` (`ModelInfo(` construction ~line 312)
- Test: `tests/test_backend_ollama.py`

**Interfaces:**
- Produces: `ModelInfo.value_length: int | None = None`, `ModelInfo.kv_lora_rank: int | None = None`, `ModelInfo.rope_dimension_count: int | None = None` (defaulted, so pre-existing constructions stay unchanged), read via `_arch_value(arch_info, "attention.value_length" | "attention.kv_lora_rank" | "rope.dimension_count")`. Task 7 consumes these names exactly.

- [ ] **Step 1: Failing test** in `tests/test_backend_ollama.py` (use the file's existing fake-show fixture pattern, the deepseek2 metadata block from `tests/data/gguf_geometry_v2/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json` plus, H-c/c′, the two new keys):

```python
def test_model_info_reports_mla_widths():
    # deepseek2: value_length 128 != key_length 192; latent dims stated
    info = <existing fake-backend helper>(DEEPSEEK2_SHOW).model_info()
    assert info.value_length == 128
    assert info.kv_lora_rank == 512
    assert info.rope_dimension_count == 64

def test_model_info_mla_fields_none_when_unstated():
    info = <existing fake-backend helper>(QWEN2_SHOW).model_info()
    assert info.value_length is None
    assert info.kv_lora_rank is None
    assert info.rope_dimension_count is None
```
(`<existing fake-backend helper>`: the file's established way of feeding a canned /api/show. Reuse it verbatim; do not invent a new fake.)
- [ ] **Step 2: Run → FAIL** (`TypeError`/`AttributeError`: unknown field). Runner as Global Constraints, output to file, `echo exit=$?`.
- [ ] **Step 3: Implement:** three defaulted fields on `ModelInfo` with a docstring line each (None = unstated, never 0; the reading is verbatim metadata, rule logic lives in geometry.py); three `_arch_value` reads in the `ModelInfo(` construction.
- [ ] **Step 4: Run → PASS**, full suite green.
- [ ] **Step 5: Commit** `feat: ModelInfo carries MLA widths (value_length, kv_lora_rank, rope.dimension_count)`.

### Task 7: geometry.py implements the rule (TDD + mutation)

**Files:**
- Modify: `src/assay/geometry.py` (`kv_bytes_per_token`, lines 153–182)
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: Task 6's field names. Produces: the same `kv_bytes_per_token(info, *, kv_bits=16) -> int | None` signature, so no caller changes.

- [ ] **Step 1: Failing tests** (branch H-b vs H-c/c′; write the winning branch only; `make_info` = the file's existing ModelInfo test helper):

```python
def test_kv_mla_separate_widths():           # H-b branch
    info = make_info(block_count=27, kv_head_count=16, head_dim=192, value_length=128)
    assert kv_bytes_per_token(info) == 276480

def test_kv_mla_latent():                    # H-c branch (H-c': expected 62208)
    info = make_info(block_count=27, kv_head_count=16, head_dim=192,
                     value_length=128, kv_lora_rank=512, rope_dimension_count=64)
    assert kv_bytes_per_token(info) == 31104

def test_kv_mla_latent_refuses_without_rank():   # H-c/c' only, R8
    info = make_info(block_count=27, kv_head_count=16, head_dim=192, value_length=128)
    assert kv_bytes_per_token(info) is None

def test_kv_equal_widths_unchanged():        # all branches: v == k is NOT MLA
    info = make_info(block_count=28, kv_head_count=8, head_dim=128, value_length=128)
    assert kv_bytes_per_token(info) == 2 * 28 * 8 * 128 * 2

def test_kv_dense_path_unchanged():          # regression: value_length None
    info = make_info(block_count=28, kv_head_count=8, head_dim=128)
    assert kv_bytes_per_token(info) == 2 * 28 * 8 * 128 * 2
```
- [ ] **Step 2: Run → FAIL** (wrong number / no branch).
- [ ] **Step 3: Implement,** inside `kv_bytes_per_token`, after the existing `parts` None-check, before the dense return. H-b:

```python
    if info.value_length is not None and info.value_length != info.head_dim:
        # R9 (MLA, separate widths): K and V stated at different widths;
        # the 2-for-K-and-V factor is replaced by the explicit sum.
        return layers * info.kv_head_count * (info.head_dim + info.value_length) * (kv_bits // 8)
```
H-c (H-c′: multiply by 2, constant `_MLA_LATENT_COPIES = 2` with a comment citing the measurement):

```python
    if info.value_length is not None and info.value_length != info.head_dim:
        # R9 (MLA, latent cache): per layer, no head factor. Rank absent
        # while MLA is stated => unmeasurable, refuse (R8), never guess.
        if info.kv_lora_rank is None or info.rope_dimension_count is None:
            return None
        return layers * (info.kv_lora_rank + info.rope_dimension_count) * (kv_bits // 8)
```
Update the function docstring: name R9, the measurement (`assay mla-kv-2026-08-27`), and that `value_length == head_dim` stays on the dense path by identity.
- [ ] **Step 4: Run → PASS**, full suite green.
- [ ] **Step 5: Mutation checks** (each: mutate, suite must FAIL naming the test, revert + `touch`, purge `__pycache__`): (1) `!=` → `==` in the guard; (2) drop `+ info.value_length` (H-b) or `+ info.rope_dimension_count` (H-c); (3) H-c: delete the rank-None refusal (must be caught by the refusal test); (4) apply the branch to the equal-widths case by removing the inequality (caught by `test_kv_equal_widths_unchanged`). Record each kill with the quoted failing assert.
- [ ] **Step 6: Commit** `feat: R9 MLA kv arithmetic per measured rule (<branch>)`.

### Task 8: Re-vendor v3 + records, then STOP for ruling

**Files:**
- Modify: `tests/data/gguf_geometry_v3/` (new vendored copy; `tests/data/gguf_geometry_v2/` deleted same commit), `tests/test_geometry_conformance.py` (docstring + set path + pinned shas + mutation-harness aim), `docs/superpowers/evidence/tier-enthusiast/ERRATA.md` (E3; skip under H-a), `src/assay/diff.py` (`SEMANTIC_BREAKS` row; skip under H-a), `docs/CARRIED-DEBT.md`, `scripts/mutate_geometry_conformance.py` (re-aim if it names v2 paths)

**Interfaces:**
- Consumes: gguf-geometry master post-Task-5 push (CI green, so vendor from the pushed commit and record its sha in the docstring).
- Produces: assay master carrying v3; the E3 erratum Task 9's bloomery work may cite.

- [ ] **Step 1: Vendor byte-exact**: `cp -a` the 11 vector files + `MANIFEST.json` from gguf-geometry `vectors/v3/` into `tests/data/gguf_geometry_v3/`; `git rm -r tests/data/gguf_geometry_v2`. Verify three ways (the v2 pattern): `cmp` file-by-file against the source checkout; `sha256sum` against `MANIFEST.json`; and the conformance test's own pinned-sha assert (updated next step).
- [ ] **Step 2: Update `tests/test_geometry_conformance.py`:** path constant to `gguf_geometry_v3`, vendored-from sha to the pushed gguf-geometry commit, docstring: v3 story (one set vendored; v2 deleted same commit; deepseek re-pinned by the measured rule; cite verdict + evidence dir). The deepseek vector now conforms through `OllamaNative.model_info()` → `geometry.py` with NO test-side mapping (the H-c/c′ metadata additions flow through the same seam). Suite must pass with the production code from Tasks 6–7. Under H-b/c/c′ the OLD code fails the new deepseek pin (run once with Tasks 6–7 stashed if cheap, to see the vector actually bites; record the red).
- [ ] **Step 3: Re-aim the mutation harness:** `scripts/mutate_geometry_conformance.py` and `test_the_mutation_harness_still_aims_at_real_lines` (tests/test_geometry_conformance.py:293): add the R9 branch line(s) as aimed lines; run the harness; all mutants killed, output to file.
- [ ] **Step 4 (skip under H-a): ERRATA.md E3.** Heading `## E3: deepseek2 MLA: kv was R2 arithmetic, not an observed allocation`, mirroring E1/E2 structure: the published figure (324 KiB/token, `source: api_show`), the measured truth (figure + lens + evidence dir), direction (over-charge → `usable_window` under-promised), affected profile files listed, corrected-value table row.
- [ ] **Step 5 (skip under H-a): SEMANTIC_BREAKS row.** First `git tag --list` / CHANGELOG check: if working `__version__` 0.13.0 is UNRELEASED, the row is `"geometry.kv_kib_per_token": (0, 13, 0)` with a comment citing R9 + E3 (this retires the standing v2-era debt item); if 0.13.0 already shipped, bump `__version__` to 0.14.0 and key the row there. Add a diff-side test in the file that covers `SEMANTIC_BREAKS` today: a profile pair straddling the row's version on that cell reads `incomparable`.
- [ ] **Step 6: CARRIED-DEBT.md.** Strike through (never delete) the retired items: "SEMANTIC_BREAKS row for geometry.kv_kib_per_token" and, if Brice so rules, items 3/4; append this slice's settled/deferred/lessons block.
- [ ] **Step 7: Full suite green** (runner + output-to-file + exit), then **STOP** for the assay merge/push ruling.

---

## Phase 3b: bloomery implements + re-vendors (repo: `~/workspace/bloomery`)

**H-a note:** Task 9 shrinks to Step 6 (doc comments) only; Task 10 runs in full.

### Task 9: gguf.rs + geometry.rs implement the rule (TDD + mutation)

**Files:**
- Modify: `crates/bloomery-core/src/gguf.rs` (GgufMeta struct ~line 15; `parse_gguf_meta` ~line 401; doc comments lines 19 and 274), `crates/bloomery-core/src/geometry.rs` (`kv_bytes_per_token` ~line 22)
- Test: `crates/bloomery-core/tests/gguf_test.rs` (key reading), geometry unit tests where `kv_bytes_per_token` is already tested

**Interfaces:**
- Produces: `GgufMeta.value_length: Option<u32>` (+ H-c/c′: `kv_lora_rank: Option<u32>`, `rope_dimension_count: Option<u32>`); same `kv_bytes_per_token(&GgufMeta) -> u64` signature. R8 refusal lives in `parse_gguf_meta` (bloomery's established refusal site), so `geometry.rs` never sees an MLA meta it cannot price.

- [ ] **Step 1: Failing tests** in `gguf_test.rs` using its existing synthetic-GGUF builder (same fixture style as the nextn_predict_layers cases): (1) deepseek2-shaped metadata (block_count 27, head_count_kv 16, key_length 192, value_length 128 [+ H-c/c′: kv_lora_rank 512, rope.dimension_count 64]) parses with the new fields populated; (2) a file WITHOUT `attention.value_length` parses with `value_length: None` (dense identity untouched); (3) H-c/c′ only: value_length ≠ key_length with kv_lora_rank ABSENT → `Err` naming the key (R8 refusal, matching the `resolve_serving_block_count` error style). Geometry test: deepseek2-shaped `GgufMeta` → `kv_bytes_per_token` == the branch figure (276480 | 31104 | 62208); an equal-widths meta (value_length == head_dim) → unchanged dense figure.
- [ ] **Step 2: `cargo test --workspace` → new tests FAIL** (output to file, `echo exit=$?`).
- [ ] **Step 3: Implement.** `lookup_u32_opt` reads in `parse_gguf_meta` (key strings `{arch}.attention.value_length`, `{arch}.attention.kv_lora_rank`, `{arch}.rope.dimension_count`); H-c/c′ refusal:

```rust
    if let Some(v) = value_length {
        if v != head_dim && (kv_lora_rank.is_none() || rope_dimension_count.is_none()) {
            return Err(GgufError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{arch}.attention.value_length {v} != key_length {head_dim} but the latent dims (attention.kv_lora_rank / rope.dimension_count) are not both present (R9/R8: MLA stated, latent size unreadable)"),
            )));
        }
    }
```
`kv_bytes_per_token` branch, H-b:

```rust
    if let Some(v) = m.value_length {
        if u64::from(v) != u64::from(m.head_dim) {
            // R9 (MLA, separate widths): explicit K+V sum replaces the
            // 2-for-K-and-V factor. Measured: assay mla-kv-2026-08-27.
            return u64::from(m.attention_layers)
                * u64::from(m.kv_heads)
                * (u64::from(m.head_dim) + u64::from(v))
                * F16_BYTES;
        }
    }
```
H-c (H-c′ multiplies by 2 via a named `MLA_LATENT_COPIES: u64` const citing the measurement): per-layer `u64::from(m.attention_layers) * (u64::from(rank) + u64::from(rope)) * F16_BYTES` where `rank`/`rope` unwrap is safe because parse refused the absent case. State that in a comment.
- [ ] **Step 4: `cargo test --workspace` → PASS.**
- [ ] **Step 5: Mutation checks** (revert + `touch`): the same four families as assay Task 7 Step 5, Rust-side; each kill recorded with the failing test name quoted.
- [ ] **Step 6: Fix the two stale doc comments,** `gguf.rs:19` and `gguf.rs:274`: drop the set-version qualifier (`gguf-geometry v1 R6` → `gguf-geometry R6 (SPEC.md)`) so the comments stop going stale per set bump. (Standing debt item: "next production-touching change", and this is it.)
- [ ] **Step 7: Commit** `feat: R9 MLA kv rule in gguf meta + geometry (<branch>); un-pin R6 doc comments from set version`.

### Task 10: Re-vendor v3 + conformance, suite, vulkan build LAST, then STOP

**Files:**
- Modify: `crates/bloomery-core/tests/data/gguf_geometry_v3/` (new; `gguf_geometry_v2/` deleted same commit), `crates/bloomery-core/tests/geometry_conformance_test.rs`

**Interfaces:**
- Consumes: gguf-geometry pushed v3 (same commit sha assay vendored), Task 9's fields.

- [ ] **Step 1: Vendor byte-exact** (cp -a + `cmp` loop + sha256 vs `MANIFEST.json`. Never restate a count without re-deriving it: the cmp loop IS the count).
- [ ] **Step 2: Update `geometry_conformance_test.rs`:** vendored-set path, source sha, header docstring v3 story; the deepseek vector now exercises the R9 path end-to-end through `parse_gguf_meta` (its metadata written into the synthetic GGUF header, H-c/c′ including the two new keys); the disproved-candidate `must_not_equal` values asserted against bloomery's own computed value (the ban-held-via-real-computation pattern from the v2 arc).
- [ ] **Step 3: `cargo test --workspace` → green** (output to file + exit).
- [ ] **Step 4: `cargo build -p bloomery-daemon --features vulkan`**, LAST, after every test run (clobber rule; featured binary must be current before any boot).
- [ ] **Step 5: Commit** `feat: vendor gguf-geometry v3; deepseek MLA conformance through parse_gguf_meta`. **STOP** for the bloomery merge/push ruling. Next-boot note stands: drift watch may flag hybrid geometry via PYTHONPATH-tracked assay. That is intended honesty; re-bless per procedure.

---

## Completion

After all rulings: SDD ledger progress appended in this repo (`.superpowers/sdd/`), memory files updated (gguf-geometry-project, oxide-robigo-convergence assay state, bloomery), and the session handoff records verdict + what moved. If the verdict was `no-match`: Tasks 4(no-match note) only, then report. An honestly open gap with a measured figure on record is a valid terminal state.
