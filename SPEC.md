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
design: K/V heads are dense in MoE architectures. Specified for
`key_length` alone — an architecture stating a different
`attention.value_length` (MLA) is a known open gap, recorded under
Versioning / v2 and not closed by this rule.

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
`serving_block_count = block_count - <arch>.nextn_predict_layers` when
the MTP key is present and nonzero. MTP layers are not serving layers.
`<arch>.nextn_predict_layers` is the GGUF spelling; HF `config.json`
calls the same quantity `mtp_num_hidden_layers`, and that name never
reaches the GGUF — the namespaced GGUF key is the one an implementation
reads, and the one the conformance vector carries. (Origin: REAP-48
prune left `mtp_num_hidden_layers: 1` in the HF config, so
`convert_hf_to_gguf` sized `block_count = 40 + 1 = 41` and emitted
`qwen35moe.nextn_predict_layers = 1` while writing 40 blocks of tensors
→ 41-vs-40 → unloadable GGUF.)

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

**v2 (2026-08-27).** The current set is `vectors/v2/`: eleven vectors —
nine of v1's ten carried forward byte-for-byte (identical shas, so the
two sets stay comparable where they overlap), two with metadata-only
additions that move no `expected` value (below), plus `qwen3.8-27b`, the
case v1 withheld. No rule text changed for v2; R1–R8 above are the rules
both sets are checked against. (The heading's `(v1)` is left as first written: this
file is amended in place and dated — see R6, amended 2026-08-27 — rather
than renumbered, and the version a consumer vendors is the directory,
per the paragraph above.)

The withheld case entered on exactly what its erratum named as closing
it: a live run charging R3/R4/R6 correctly, by the corrected instrument
at assay master `24420f2`, captured in assay
`docs/superpowers/evidence/qwen38-27b-live-2026-08-27/` and committed at
assay `8892d0e`. Pinning its published 260 KiB/token figure in v1 would
have frozen a 4.0625x attention over-charge *as the contract*, so it
waited for a later set rather than entering v1 as an unverified number
(`docs/upstream-errata/2026-08-27-assay-qwen3.8-27b-hybrid-overcharge.md`).
Two carried-forward vectors additionally state metadata keys their v1
copies cited but omitted — gemma-4's sliding-window/`*_swa` family and
deepseek-coder-v2's `attention.value_length` — and neither `expected`
block moved: completing the stated inputs is not a re-derivation.

**Known open gap — R2 and MLA (recorded 2026-08-27, not closed).** R2 is
specified for `key_length` alone: R1's `head_dim` is the key width, and
the factor 2 stands for K and V. That is the arithmetic behind every kv
figure in both sets, including deepseek-coder-v2's published,
hardware-verified `kv_bytes_per_token` 331776. What v2 makes visible is
the case that reading does not cover: `vectors/v2/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`
now states `attention.value_length` 128 against `attention.key_length`
192 (MLA — two different head widths), so an implementation that models
them separately derives a different number and fails that vector. No rule
here settles which is right, and this repo derives nothing, so the gap is
recorded rather than closed: closing it needs a rule and a measurement.
Until both exist, R2 means what it says, and the vector pins the measured
figure.

`vectors/v1/` stays frozen and valid. Its withheld-case record remains
true of v1 forever; v2 is where the closure lands.
