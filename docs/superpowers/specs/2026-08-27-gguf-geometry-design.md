# gguf-geometry: design spec

Date: 2026-08-27
Status: draft, awaiting Brice's review
Origin: the 2026-08-27 portfolio review (external Grok analysis, item A,
adopted with modifications ruled in conversation).

## 0. Purpose

Three repos independently implement GGUF metadata interpretation and the
context-window law, and each lineage has produced a measured defect:

| Repo | Implementation | Measured defect |
|------|----------------|-----------------|
| assay | `src/assay/geometry.py` (Python) | E1 erratum: `embedding // head_count` head_dim derivation over-promised usable windows 17–33% on 6 of 23 committed profiles (both signs exist: mistral-nemo under-promised) |
| bloomery | `crates/bloomery-core/{gguf,geometry}.rs` (Rust) | Hybrid KV derived from all 40 blocks instead of `full_attention_interval` (4× over-charge on Qwen3.6-35B-A3B: 40 blocks charged, 10 real attention layers); DeltaNet recurrent state (62.8 MiB/ctx) uncharged; both fixed in turn 5 |
| robigo | `src/robigo/model/{gguf,geometry}.py` (Python) | Origin of the window law; frozen as a findings repo, not a consumer of this contract |

A related parse-layer trap is on record: REAP-pruned GGUFs carrying
`mtp_num_hidden_layers: 1` yield a 41-vs-40 block count and an unloadable
model until the key is handled (bloomery CARRIED-DEBT, 2026-08-22).

This repo exists so the *same* interpretation bug can never again live
silently in two implementations at once. It pins one written contract and
one frozen set of conformance vectors; each consumer proves against them
in its own CI.

**Defining property: this repo contains no geometry implementation.**
It is contract + evidence only. Expected values in vectors come from
already-published, already-verified evidence, never from code in this
repo. A reference implementation here would be a third implementation
with a fourth bug.

## 1. Non-goals

- **No implementation.** No library, no bindings, no reference
  calculator. (Rejected alternatives: a Rust crate with Python bindings
  breaks assay's stdlib-only property and recreates the live-coupling
  failure measured in the 2026-08-18 assay-v1.8/bloomery PYTHONPATH
  incident; canonicalizing in assay homes the contract in a fast-revving
  instrument.)
- **No endpoint or box-state geometry.** VRAM-at-boot readings, daemon
  context reporting, and residency. Those are backend-specific and stay in each
  instrument.
- **No binary GGUF reader spec in v1.** Every measured bug was
  key-*interpretation*, not byte-parsing. Vectors take extracted
  metadata as input. If a reader-level divergence ever appears, binary
  fixtures join a later vector version.
- **No robigo changes.** Robigo is frozen as a findings repo.
- **Not a unified framework.** One narrow contract; consumers keep their
  own code, release cadence, and instruments.

## 2. The contract (SPEC.md)

SPEC.md states, with one canonical rule each:

1. **head_dim derivation**: `attention.key_length` preferred; the
   fallback chain and when `embedding // head_count` is admissible
   (the E1 rule).
2. **KV bytes per token**: the formula and its inputs (heads_kv,
   head_dim, layer count, cache dtype bytes), stated per-term.
3. **Attention layer count for hybrid models**: derived via
   `full_attention_interval` (or the architecture's equivalent), never
   the raw block count.
4. **Recurrent state**: a distinct per-context charged term
   (`recurrent_state_bytes`), derived from `ssm.*` keys; zero only when
   the architecture has no recurrent layers.
5. **MoE expert fields**: absent means *absent* (`null`/`None`), never
   `0`. Expert count and experts-used are separate fields.
6. **Block count**: the `mtp_num_hidden_layers` rule. MTP layers are
   not serving layers.
7. **The window law**: a pure function. Inputs: a named term list
   (budget bytes, weights bytes, kv bytes/token, per-context fixed
   overhead, recurrent bytes, any consumer-declared extra terms, each
   named). Outputs: `usable_window` and `limited_by` (which term bound
   it). Consumers map their own config onto the named terms; the law
   itself has no configuration.
8. **Unknown architecture**: refuse / mark unknown. Guessing is a
   contract violation, not a fallback.

**Versioning**: the vector-set directory (`vectors/v1/`, `v2/`, …) is
the contract version. Any semantic change to a rule, not just new
models, bumps the set. Old sets are never edited (same discipline as
assay's semantic-break registry: comparisons across a break are
incomparable, not wrong).

## 3. Vector format

`vectors/v1/<model-id>.json`, one per model:

```json
{
  "id": "qwen3.6-35b-a3b-reap48-ours-q4km",
  "source": {
    "gguf_sha256": "90e2181e...",
    "provenance": "bloomery docs/superpowers/evidence/2026-08-22-g5v4-reap48-baselines.md"
  },
  "metadata": { "...": "geometry-relevant GGUF keys, verbatim" },
  "expected": {
    "kv_bytes_per_token": 20480,
    "attention_layers": 10,
    "recurrent_state_bytes": 65863680,
    "experts": { "count": 133, "used": 8 },
    "windows": [
      { "terms": { "budget_bytes": 0, "weights_bytes": 0, "fixed_overhead_bytes": 0 },
        "usable_window": 0, "limited_by": "kv" }
    ]
  },
  "must_not_equal": {
    "kv_bytes_per_token": [81920],
    "note": "all-40-blocks derivation, fixed bloomery turn 5"
  },
  "verification": "hardware-verified | derived"
}
```

(Numbers above are illustrative of *shape*; every real value enters only
by citation. Window scenarios state their term values explicitly so the
law (not box state) is what's tested.)

Rules:

- **Provenance is mandatory.** Every `expected` block cites the evidence
  doc it comes from and carries `hardware-verified` (a live run measured
  it) or `derived` (computed in a published, reviewed doc). No number
  without a source.
- **`must_not_equal` pins historical wrong answers** as adversarial
  cases: the old head_dim formula's numbers on the six E1-affected
  models, the 40-block figure on the hybrid, the 41-block count on the
  mtp case. A regressed implementation must fail these by construction.
- **`MANIFEST.json`** lists every vector file with its sha256 and the
  set's contract version. Consumers assert the manifest sha of their
  vendored copy.

## 4. Seed corpus (v1)

- **The E1 sweep** (assay `docs/superpowers/evidence/e1-sweep/`):
  23 profiles with raw `show-*.json` metadata, pre-registered protocol,
  corrected derivations, and four values hardware-verified in the v1.7
  campaign. Covers 5 of the sweep's 6 affected models: 4 carrying a
  corrected kv plus the old value as `must_not_equal` (both signs:
  mistral-nemo under-promised, the rest over-promised), and gemma-4,
  which is affected *and* is the geometry:null edge, entering as the R8
  refusal, plus 3 unaffected controls.
  **Withheld (ruled 2026-08-27): `qwen3.8:27b`.** Its published 260
  KiB/token charges all 65 raw blocks against metadata that states
  `full_attention_interval`, `nextn_predict_layers` and `ssm.*`, so
  R3/R4/R6 all apply and pinning the published figure would freeze a
  4.06x over-charge *as the contract*. Deriving the conforming value
  here is banned (section 0), so the case waits for a conforming
  measurement or an upstream correction. Filed as
  [`docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md`](../../upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md);
  pinned by `tests/test_v1_vectors.py`. The E1 corpus alone therefore
  exercises R1/R2/R5/R7/R8; R3/R4/R6 arrive with the hybrid below.
- **The hybrid case**: Qwen3.6-35B-A3B REAP-48 (bloomery turn-5
  evidence: `attention_layers` 10, kv 20,480 B/tok, recurrent
  65,863,680 B, hardware-verified across two boots).
- **The mtp trap**: REAP-48-ours pre-patch metadata (41-vs-40 blocks) as
  a `must_not_equal` on block count.
- **A real MoE**: deepseek-coder-v2:16b-lite (64/6 expert fields, from
  the assay tools-anchor / E1 evidence).

Target: every vector in v1 is traceable to committed evidence in assay
or bloomery. If a desired case has no committed evidence, it waits for a
later set rather than entering as an unverified number.

## 5. Checker CLI (`tools/check.py`)

Stdlib-only Python. Takes an implementation's emitted results file
(JSON: per-model derived values, shape mirroring `expected`) and a
vector directory; compares field-by-field; reports per-model pass/fail
with the differing terms; exit 0 all pass / 1 any fail / 2 usage or
malformed input. It computes no geometry. Consumers may use it or write
their own comparison in-language (bloomery likely compares in Rust; the vendored vectors are the contract, the checker is a convenience).

## 6. Consumer integration

Ordered, with an interference guard recorded from the 2026-08-27 box
survey (bloomery memory-battery worktree live and GPU-gated; crucible
Pillar-1 running on a rental; both untouched by this work):

1. **This repo first** (no shared state with anything in flight):
   SPEC.md, vectors/v1, checker, CI (vector JSON shape + manifest shas +
   checker unit tests, mutation-checked per house rule).
2. **assay second, on a branch in its own worktree.** Vendor the
   sha-pinned vector set, add one conformance test module driving
   `geometry.py` over it. **Predicted red**: `geometry.py` (132 lines)
   has no hybrid/recurrent handling, so the REAP-48 vector should fail;
   the fix is a real assay geometry change made under assay's normal
   discipline (CHANGELOG; errata if committed profiles are affected;
   schema/semantic-break review if profile fields change).
   **MERGE HOLD**: nothing merges to assay master while the bloomery
   memory-battery is in flight: bloomery's deployed assay follows a
   PYTHONPATH pin, and a mid-battery geometry change on master risks
   changing the battery's probe layer in place (the measured v1.8
   failure mode). The branch waits; the battery gate releases it.
3. **bloomery third, only after the memory-battery arc closes**:
   vendored copy + conformance test in bloomery-core. Test-only, but no
   second arc opens in that repo while the battery worktree is live.
4. **robigo: never.** Frozen.

## 7. Acceptance

- Repo CI green: vector shape validation, manifest shas, checker tests
  (mutation-checked).
- Both consumers' conformance suites green against v1.
- At least one previously-silent divergence surfaced and resolved
  through the spec (the assay hybrid gap is the expected instance).
  If the corpus surfaces zero divergences, that is reported plainly as
  "no divergence found on this corpus", not claimed as proof of
  equivalence.

## 8. Repo facts

Public GitHub repo under bricelancasterwcp-sudo, MIT, `~/workspace/`
checkout, matching house defaults. Secret-scan before any push; verify
origin sync after every push. Publish only after Brice approves this
spec.
