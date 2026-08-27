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
