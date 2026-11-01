# MLA KV Rule: Measurement + v3 Design (2026-08-27)

Closes (or honestly declines to close) the SPEC's "Known open gap: R2
and MLA": deepseek2 states `attention.value_length` 128 against
`attention.key_length` 192, R2 is specified for `key_length` alone, and
no measurement has ever observed what the runtime actually allocates.
Approved scope: **full chain:** measurement → rule + `vectors/v3` →
both consumers implement and re-vendor. Approach: **pre-registered
one-off protocol** (assay-owned evidence), rule written only to the
measured outcome.

## 0. Ground rules

- This repo derives nothing (design spec 2026-08-27 §0). The rule and
  vectors enter only by citation of assay-committed evidence.
- The pre-registration instrument of record is the PROTOCOL.md in
  assay (below); this design describes it and binds its commitments.
  If the two ever disagree, the committed protocol wins.
- The point estimate decides. No re-run, no candidate added, no band
  widened after a number is seen.
- Merges and pushes happen only on Brice's explicit per-phase rulings.

## 1. Phase 1: the measurement (assay)

**Evidence dir:** assay `docs/superpowers/evidence/mla-kv-<capture-date>/`
(dated at capture). `PROTOCOL.md` is written and committed **before any
instrument runs**.

**Subject and lens (named):** ollama 0.32.13 on this box, model
`deepseek-coder-v2:16b-lite-instruct-q5_K_M` (store digest asserted
against manifest model-layer digest `bc286970…2406` before measuring),
KV cache type f16 (`OLLAMA_KV_CACHE_TYPE` unset; env recorded),
`OLLAMA_FLASH_ATTENTION` and `OLLAMA_NUM_PARALLEL` recorded, runner
(new engine vs llama runner) recorded from logs, timestamps in
provenance. Point-in-time capability of this runtime, not a property
of the GGUF.

**Candidates (arithmetic shown; all inputs verbatim GGUF keys:
block_count 27 dense attention so attention_layers = 27, head_count_kv
16, key_length 192, value_length 128, kv_lora_rank 512,
rope.dimension_count 64, f16 = 2 B/elt):**

| Id | Model of the cache | Formula | B/token |
|----|--------------------|---------|---------|
| H-a | k-width for both K and V (current pin) | 2·27·16·192·2 | 331,776 |
| H-b | separate K and V widths | 27·16·(192+128)·2 | 276,480 |
| H-c | MLA latent, per-layer (no head factor) | 27·(512+64)·2 | 31,104 |
| H-c′ | MLA latent + transposed copy | 2·27·(512+64)·2 | 62,208 |

H-c′ is included because llama.cpp's MLA implementation has historically
cached a transposed copy of the latent for decode speed; it is a derived
implementation variant, not a hedge added after data.

**Decision rule (pre-registered):** instrument 1's bytes/token matching
exactly one candidate within ±5% decides it. Sanity check on the chosen
band: the closest candidate pair (H-a vs H-b) is 20% apart and every
other pair is ≥2×, so ±5% bands cannot overlap. The band is generous
without being ambiguous. **No match → no rule is written**; the
measured figure is recorded, the SPEC gap stays open, Phases 2–3 do
not run.

**Instrument 1 (primary, log lines):** load the model with an explicit
`num_ctx`; quote verbatim the llama.cpp init lines (KV buffer size and
the `K (f16): … V (f16): …` split, which alone already
distinguishes H-a from H-b, and a latent cache reports a shape neither
predicts). bytes/token = reported KV bytes ÷ the **n_ctx the log
reports** (not the requested one; parallel slots inflate the effective
n_ctx and the log's own figure is the denominator of record). Every
"because" in the record is a quoted log line.

**Instrument 2 (corroboration, VRAM slope):** load at
num_ctx ∈ {2048, 4096, 8192}; record per-process VRAM each time.
Validity condition per point: `ollama ps` reports 100% GPU, else the
point is named in `dropped` (CPU-split KV is invisible to nvidia-smi).
Known confound, stated up front: compute buffers also grow with ctx, so
the slope over-estimates KV per token. Pre-registered consequence:
instrument 2 **cannot decide alone**; it must only be *consistent* with
instrument 1 (slope ≥ instrument 1's figure, same order of magnitude).
GPU hygiene: no competing models loaded; free VRAM recorded before each
point.

**Pre-registered contingency:** if ollama 0.32.13's logs do not surface
usable KV-buffer lines, the fallback is the same blob under a standalone
`llama-server`, a **different named lens**. Whether a rule may be
written from that lens is Brice's ruling at that moment; it is not a
silent swap.

**Infrastructure vs measurement:** a failed load, OOM, or missing log
line is infrastructure, recorded as such; it is never data and never a
re-roll of a completed reading.

## 2. Phase 2: rule + vectors/v3 (gguf-geometry)

**Outcome → rule mapping (pre-registered):**

- **H-a** → R2 amended in place, dated (R6 pattern): under the measured
  runtime, `value_length` does not affect cache sizing; gap closed by
  confirmation.
- **H-b** → new **R9 (MLA, separate widths)**: when
  `attention.value_length` is present and ≠ `key_length`,
  `kv_bytes_per_token = attention_layer_count · head_count_kv ·
  (key_length + value_length) · bytes_per_element`; R2 defers to R9 in
  that case.
- **H-c or H-c′** → new **R9 (MLA, latent cache)**:
  `kv_bytes_per_token = attention_layer_count · (kv_lora_rank +
  rope.dimension_count) · bytes_per_element` (× 2 iff H-c′ is the
  measured truth; the factor enters only by measurement), per layer,
  no head factor. R8 refusal when `value_length ≠ key_length` but
  `attention.kv_lora_rank` is absent.
- **No match** → no rule, no v3; the SPEC open-gap paragraph gains the
  measured figure with its lens named; arc stops and reports.

The winning rule carries an origin note scoped to what was measured:
one architecture (deepseek2), one runtime (ollama 0.32.13), this box,
dated, the R1/R3/R6 origin pattern.

**vectors/v3 (cut on any decisive outcome):**

- Ten vectors carried forward **byte-for-byte** (identical shas; sets
  comparable where they overlap, the v2 pattern). Comparisons across sets
  remain incomparable-not-wrong.
- The deepseek vector's `expected.kv_bytes_per_token` becomes the
  **winning formula's exact output** (the measurement verifies it
  within band; the formula supplies the byte figure. A pinned value is
  never a rounded reading). Its `metadata` gains
  `deepseek2.attention.kv_lora_rank` and
  `deepseek2.rope.dimension_count` (verbatim from the committed show
  capture) iff the winning rule consumes them.
- `must_not_equal` gains the disproved candidates, **including 331,776
  itself if H-a loses**, with a note naming the erratum, alongside the
  existing 221,184 E1 pin. A regressed implementation fails by
  construction.
- v1 and v2 stay frozen and valid. SPEC gains a dated **v3** paragraph;
  the "Known open gap" text is amended non-silently: the old claim
  stays visible, closure appended with date (withdrawn-claims
  discipline).
- If H-a loses: new upstream-errata doc
  (`docs/upstream-errata/<date>-assay-deepseek-mla-kv-overcharge.md`,
  E3-style) stating plainly that assay's published "hardware-verified"
  324 KiB/token was R2 arithmetic executed on a live box
  (`geometry.source: api_show`), never an observed allocation, with
  direction (over-charge → windows under-promised).

**Mechanics:** `tests/test_v3_vectors.py` mirrors the v2 test;
`MANIFEST.json` regenerated; validator only changes if the vector shape
needs it (none anticipated, since the must_not_equal key guard exists); CI
green before any consumer touches v3.

## 3. Phase 3: consumers

**assay (3a):** implement the winning rule in the geometry computation
(the `api_show`/model_info path): TDD red-first, mutation checks on
the new branch logic. ERRATA.md gains E3. The standing SEMANTIC_BREAKS
debt for `geometry.kv_kib_per_token` folds into this change (the
semantic actually moves now). Re-vendor v3 (v2 dir deleted; consumers
vendor one current set), byte-exactness asserted three ways as in the
v2 re-vendor, mutation harness re-aimed. Published tier profiles are
not retro-edited; the erratum is the record (E1/E2 pattern).

**bloomery (3b):** `parse_gguf_meta` implements the rule; contract test
moves to v3 and must hold the new pins with real computed values. The
two stale "gguf-geometry v1 R6" doc comments in `src/gguf.rs` are fixed
in the same change (the debt item queued for "next production-touching
change"). Featured vulkan binary rebuilt **last** (clobber rule).

## 4. Order, gates, environment

1. Phase 1: protocol commit → capture → evidence commit (assay).
   Gate: Brice reads the verdict; rules on proceeding.
2. Phase 2: gguf-geometry rule + v3, CI green. Gate: merge/push ruling.
3. Phase 3a assay, 3b bloomery (independent once v3 is live; 3b after
   3a only if a shared ruling says so). Gates: per-repo merge/push
   rulings.

Environment (handoff-pinned): assay/gguf-geometry test runner
`PYTHONDONTWRITEBYTECODE=1 uv run --offline --with pytest --python 3.12
python -m pytest -q`; bloomery `cargo test --workspace` then ALWAYS the
vulkan featured build last; full output to a file + `echo exit=$?` on
every verification run; mutation testing reverts then `touch`. SDD
ledger in gguf-geometry `.superpowers/sdd/` per house pattern.

## 5. Non-goals

- No standing observed-allocation instrument in assay (Approach B,
  declined, available as a future arc).
- No rule for MLA architectures beyond what the deepseek2 measurement
  grounds; no second-model generality claim.
- No retro-editing of published assay profiles or frozen vector sets.
- No gemma-4 non-swa extra keys, no per-vector mutant naming. Those are separate
  queued items, untouched here.
