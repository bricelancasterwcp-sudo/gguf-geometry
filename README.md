# gguf-geometry

One contract for GGUF metadata interpretation and the context-window law,
so the same interpretation bug can never again live silently in two
implementations at once.

## Why this repo has no implementation

**This repo contains no geometry implementation.** It is contract plus
evidence, nothing else. A reference implementation living here would just
be a third implementation with a fourth bug — one more place the same
mistake could hide. Instead, this repo pins one written contract
([SPEC.md](SPEC.md)) and frozen sets of conformance vectors (`vectors/v2/`
is current, `vectors/v1/` frozen); each consumer proves its own
implementation against them in its own CI.

## The problem this fixes

Three lineages have each implemented GGUF metadata interpretation
independently, and each has produced a measured defect:

| Repo | Implementation | Measured defect |
|------|----------------|------------------|
| assay | `src/assay/geometry.py` (Python) | E1 erratum: `embedding // head_count` head_dim derivation over-promised usable windows 17–33% on 6 of 23 committed profiles (both signs exist: mistral-nemo under-promised) |
| bloomery | `crates/bloomery-core/{gguf,geometry}.rs` (Rust) | Hybrid KV derived from all 40 blocks instead of `full_attention_interval` (4× over-charge on Qwen3.6-35B-A3B — 40 blocks charged, 10 real attention layers); DeltaNet recurrent state (62.8 MiB/ctx) uncharged; both fixed in turn 5 |
| robigo | `src/robigo/model/{gguf,geometry}.py` (Python) | Origin of the window law; frozen as a findings repo — not a consumer of this contract |

A related parse-layer trap is on record: a REAP-pruned checkpoint whose HF
`config.json` carries `mtp_num_hidden_layers: 1` converts to a GGUF carrying
`<arch>.nextn_predict_layers: 1` and `block_count` 41 — a 41-vs-40 block
count and an unloadable model until the key is handled (SPEC.md R6; the
GGUF spelling is the one an implementation reads).

## The contract

[SPEC.md](SPEC.md) states eight rules, R1–R8, covering head_dim
derivation, KV bytes per token, hybrid-model attention layer counting,
recurrent state, MoE expert fields, MTP-aware block counting, the window
law itself, and the refusal behavior for unknown architectures. Each rule
cites the measured defect it closes.

**Versioning:** the vector-set directory (`vectors/v1/`, `v2/`, ...) is
the contract version. Any semantic change to a rule bumps the set. Old
sets are never edited — comparisons across sets are incomparable, not
wrong.

## Conformance vectors

`vectors/<set>/<model-id>.json` — one file per model, each stating the
metadata GGUF keys and the expected geometry values a conforming
implementation must derive from them, plus (where applicable)
`must_not_equal`: historical wrong answers pinned as adversarial cases.

**Current set: `vectors/v2/`** (2026-08-27) — eleven vectors: v1's ten
carried forward byte-for-byte, plus `qwen3.8-27b`, the case v1 withheld,
which entered on the live conforming run that closed its erratum.
`vectors/v1/` (ten vectors) stays frozen and is never edited.

**Expected values enter only by citation from published evidence.** Every
vector's `expected` block names the evidence document it comes from in
`source.provenance` and states `verification` as `hardware-verified` (a
live run measured it) or `derived` (computed in a published, reviewed
doc). No number in this repo is computed here — copying a number into a
vector is not deriving it.

**A case with no honest number stays out, on the record.** When published
evidence for a model contradicts SPEC.md, the vector is withheld rather
than pinned at the contradicting value — pinning it would freeze the
defect as the contract. Withheld cases are filed in
[`docs/upstream-errata/`](docs/upstream-errata/) with the evidence, the
rule they conflict with, and what would close them; the test suite pins
both the absence and its record. When a case closes, it lands in the next
set rather than being back-fitted into the frozen one: `qwen3.8-27b` was
withheld from v1 and enters v2 on the live conforming measurement its
erratum named as the thing that would close it.

## Consumer model

1. Vendor a sha-pinned copy of one set — `vectors/v2/` for new work — into
   your own repo.
2. Assert the vendored `MANIFEST.json` sha in your test suite, so a
   silent re-vendor is a visible diff.
3. Run your own implementation over the vectors in your own CI — this
   repo does not run anyone's implementation for them.

## Tools

Stdlib-only Python 3.12+, no third-party runtime dependencies (pytest is
a dev dependency for this repo's own tests only):

- `tools/validate_vectors.py` — validates a vector directory's shape and
  manifest against the schema in SPEC.md / §3 of the design spec. Computes
  no geometry.
- `tools/gen_manifest.py` — writes `MANIFEST.json` (sha256 per vector
  file) for a vector directory.
- `tools/check.py` — compares a consumer's emitted results against a
  vector directory, field by field, and fails on any `must_not_equal`
  hit. Computes no geometry; it only compares.

## Running the checks

These commands check this repo, and CI is held to the same ones — one
validation per set, because a frozen set is only frozen while something
still checks it:

```bash
python -m pytest tests/ -q                   # the pinning suite
python tools/validate_vectors.py vectors/v1  # vector shape + manifest; exit 0 = clean
python tools/validate_vectors.py vectors/v2  # the current set, same check
```

Nothing beyond the standard library and pytest is needed, so
`python -m pip install pytest` on Python 3.12+ is the whole environment.
`tests/test_ci_contract.py` pins that pairing: it asserts this section
documents the suite and the vector validation, and — as soon as a workflow
exists under `.github/workflows/` — that CI runs the same commands against
the same vector directory, on a Python it pins at 3.12 or newer.
`tests/test_v2_vectors.py` extends the same pin to the current set, so a
new set cannot be published with nothing checking it. Documented check and
enforced check cannot drift apart silently.

## Non-goals

No binary GGUF reader (vectors take extracted metadata as input — every
measured bug so far was key-interpretation, not byte-parsing), no
endpoint or box-state geometry, no unified framework. See
`docs/superpowers/specs/2026-08-27-gguf-geometry-design.md` for the full
design rationale.
