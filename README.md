# gguf-geometry

One contract for GGUF metadata interpretation and the context-window law,
so the same interpretation bug can never again live silently in two
implementations at once.

## Why this repo has no implementation

**This repo contains no geometry implementation.** It is contract plus
evidence, nothing else. A reference implementation living here would just
be a third implementation with a fourth bug — one more place the same
mistake could hide. Instead, this repo pins one written contract
([SPEC.md](SPEC.md)) and one frozen set of conformance vectors
(`vectors/v1/`); each consumer proves its own implementation against them
in its own CI.

## The problem this fixes

Three lineages have each implemented GGUF metadata interpretation
independently, and each has produced a measured defect:

| Repo | Implementation | Measured defect |
|------|----------------|------------------|
| assay | `src/assay/geometry.py` (Python) | E1 erratum: `embedding // head_count` head_dim derivation over-promised usable windows 17–33% on 6 of 23 committed profiles (both signs exist: mistral-nemo under-promised) |
| bloomery | `crates/bloomery-core/{gguf,geometry}.rs` (Rust) | Hybrid KV derived from all 40 blocks instead of `full_attention_interval` (4× over-charge on Qwen3.6-35B-A3B — 40 blocks charged, 10 real attention layers); DeltaNet recurrent state (62.8 MiB/ctx) uncharged; both fixed in turn 5 |
| robigo | `src/robigo/model/{gguf,geometry}.py` (Python) | Origin of the window law; frozen as a findings repo — not a consumer of this contract |

A related parse-layer trap is on record: REAP-pruned GGUFs carrying
`mtp_num_hidden_layers: 1` yield a 41-vs-40 block count and an unloadable
model until the key is handled.

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

`vectors/v1/<model-id>.json` — one file per model, each stating the
metadata GGUF keys and the expected geometry values a conforming
implementation must derive from them, plus (where applicable)
`must_not_equal`: historical wrong answers pinned as adversarial cases.

**Expected values enter only by citation from published evidence.** Every
vector's `expected` block names the evidence document it comes from in
`source.provenance` and states `verification` as `hardware-verified` (a
live run measured it) or `derived` (computed in a published, reviewed
doc). No number in this repo is computed here — copying a number into a
vector is not deriving it.

## Consumer model

1. Vendor a sha-pinned copy of `vectors/v1/` into your own repo.
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

## Non-goals

No binary GGUF reader (vectors take extracted metadata as input — every
measured bug so far was key-interpretation, not byte-parsing), no
endpoint or box-state geometry, no unified framework. See
`docs/superpowers/specs/2026-08-27-gguf-geometry-design.md` for the full
design rationale.
