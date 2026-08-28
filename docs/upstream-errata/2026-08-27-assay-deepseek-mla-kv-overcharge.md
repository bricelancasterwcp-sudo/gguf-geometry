# Upstream erratum — assay's published `deepseek-coder-v2:16b-lite-instruct-q5_K_M` KV geometry

Date filed: 2026-08-28
Filed by: gguf-geometry Task 4 (MLA KV rule closure), recorded here
because this repo cannot write to assay.
Status: **open** — filed against assay, not yet corrected upstream.
Effect on this repo: `vectors/v1/` and `vectors/v2/` keep their
published `deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`
`expected.kv_bytes_per_token` **331776 unchanged** (old sets are never
edited — see SPEC.md Versioning); `vectors/v3/` re-pins the vector to
the measured figure under the new rule, R9.

## Summary

assay publishes `deepseek-coder-v2:16b-lite-instruct-q5_K_M` at **324
KiB/token** (331,776 B/token) of KV cache, marked "hardware-verified."
That figure is SPEC.md R2's arithmetic — `2 * layers * head_count_kv *
head_dim * bytes_per_element` with R1's `head_dim = key_length = 192`
— applied to a model that is MLA: the same metadata states
`attention.value_length 128`, a *different* width from
`attention.key_length 192`, which R2 was never specified to cover
(SPEC.md's "Known open gap — R2 and MLA," recorded 2026-08-27). A live
measurement taken 2026-08-27 under ollama 0.32.13 (Task 2/3 of this
plan; assay `docs/superpowers/evidence/mla-kv-2026-08-27/`) shows the
runtime does not allocate the R2 figure at all: it allocates **separate
K and V widths**, 276,480 B/token — 1.2x smaller than the published
number. The published figure over-charges the KV term, which
under-promises the usable context window it feeds into R7.

## What was published

| Where | Value |
|---|---|
| `vectors/v1/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json` (this repo) | `expected.kv_bytes_per_token: 331776`, `verification: "hardware-verified"` |
| `vectors/v2/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json` (this repo) | same `331776`, plus `attention.value_length 128` stated beside `attention.key_length 192` without moving `expected` |
| assay `docs/superpowers/evidence/tier-enthusiast/ERRATA.md`, erratum E1 ("Confirmed affected profiles" table) | committed `216` -> corrected `324` KiB/token |
| assay `docs/superpowers/evidence/e1-sweep/results.json`, profile `tier-enthusiast/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json` | `corrected.kv_kib_per_token: 324` |
| assay `docs/superpowers/evidence/tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json` (probe 0.9.0) | `geometry.kv_kib_per_token: 324`, `geometry.source: api_show` |

## What it actually was

`331,776 = 2 * 27 * 16 * 192 * 2` — R2 arithmetic, run on
`attention.key_length` alone, on a live box. It was never an *observed*
allocation: the probe's own field says so —
`geometry.source: api_show` in
`tier-enthusiast-2026-08/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`
names the metadata read (`/api/show`), not a KV buffer measured from a
running load. "Hardware-verified" in this repo's vector metadata refers
to the *model identity and metadata* being read from real hardware, not
to the KV figure being an observed cache size — a distinction this
erratum exists to make explicit, because nothing before Task 2 of this
plan had loaded the model and read its actual KV buffer allocation.

## The measured figure and its lens

Task 2 captured three ollama 0.32.13 loads of this exact blob (sha256
`bc286970a24072cf23a4c905f28adb9f6a28c71743b07790185275a86dc72406`) at
`num_ctx` 2048, 4096 and 8192; Task 3 applied the pre-registered
decision rule from `PROTOCOL.md` (committed c379c9f, before the
captures in cf6222e). Full arithmetic, verbatim quoted log lines:
assay `docs/superpowers/evidence/mla-kv-2026-08-27/READING.md`;
verdict: `VERDICT.md` in the same directory.

**Instrument 1 (server-log KV buffer lines), all three points, exact
and identical:**

```
276,480 bytes/token
  = 27 layers * 16 heads * (192 + 128) * 2 bytes/element (f16)
  = 27 * 16 * 320 * 2
```

| point | `CUDA0 KV buffer size` | log-reported n_ctx | bytes/token |
|---|---|---|---|
| ctx 2048 | 540.00 MiB | 2048 | 540 × 1,048,576 ÷ 2048 = 276,480 |
| ctx 4096 | 1080.00 MiB | 4096 | 1080 × 1,048,576 ÷ 4096 = 276,480 |
| ctx 8192 | 2160.00 MiB | 8192 | 2160 × 1,048,576 ÷ 8192 = 276,480 |

The log's own K/V split corroborates the two widths separately, digit
for digit, at every point: K 165,888 = `27*16*192*2`, V 110,592 =
`27*16*128*2`; 165,888 + 110,592 = 276,480. Instrument 2 (VRAM-slope
corroboration, cannot decide alone) is consistent — same order of
magnitude, slope ≥ Instrument 1 on every segment, the excess fully
accounted for by growing compute buffers (the pre-registered confound).
Four candidates were banded at ±5% before either capture; exactly one
matched: H-b, deviation 0.000%.

**Lens — this is a property of one runtime on one box, not of the
GGUF:**

- **Non-flash-attention reading.** `flash_attn = auto` resolved to
  Flash Attention assigned to device CPU on all three loads
  (`resolve_fused_ops: ... Flash Attention is assigned to device CPU`,
  uniform across all three logs) — this is a reading of ollama
  0.32.13's non-FA path on an RTX 5080. A future FA-enabled build could
  read differently; nothing here claims otherwise.
- **The blob is MLA; the runtime does not cache the latent.** The same
  logs state `deepseek2.attention.kv_lora_rank 512` /
  `print_info: n_lora_kv 512` — this model *is* MLA — yet the measured
  runtime materializes full per-head K (192-wide) and V (128-wide)
  rather than caching the 512+64 latent representation. A rule that
  keyed on "the architecture is MLA" and assumed a latent cache would
  have derived a per-layer, no-head-factor figure roughly **8.9x**
  smaller than what was actually allocated (276,480 ÷ 31,104 ≈ 8.89).
  That is why SPEC.md's new rule, R9, keys on the stated
  `attention.value_length != attention.key_length` fact alone, never on
  the architecture name.
- Runner: ollama's own llama-server subprocess (`source=server.go:109
  msg="using llama-server for model"`), not a standalone llama-server —
  the protocol's primary lens, not its contingency.
- One box (brice-X870E-Taichi, RTX 5080), one model
  (`deepseek-coder-v2:16b-lite-instruct-q5_K_M`,
  digest `sha256:bc286970...`), captured 2026-08-27
  23:56:36–23:56:52 CDT.

## Direction

**Over-charge.** The published figure (331,776 B/token) charges more
KV per token than the runtime actually allocates (276,480 B/token) —
a factor of `331776 / 276480 = 1.2x`. Fed through R7's window law, an
over-charged `kv_bytes_per_token` produces a smaller `usable_window`
than the box can actually serve: published windows for this model were
**under-promised**, not the over-promise direction R1's erratum and the
qwen3.8-27b erratum both recorded. This is the opposite sign from those
two priors, recorded rather than assumed.

## Affected artifacts

- **assay's published tier profiles** carrying 324 KiB/token for this
  model (`tier-enthusiast/ERRATA.md` E1 row, the `e1-sweep` corrected
  figure, the `tier-enthusiast-2026-08` probe-0.9.0 row, and any matrix
  page built from them) — assay's call to correct, not this repo's.
- **`vectors/v1/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`** —
  **frozen, unchanged.** SPEC.md Versioning: "Old sets are never
  edited; comparisons across sets are incomparable, not wrong." v1's
  `expected.kv_bytes_per_token` stays 331776 forever; this erratum is
  the record of what that number actually was, not an edit to it.
- **`vectors/v2/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`** —
  **frozen, unchanged**, same reasoning. v2 additionally stated
  `attention.value_length` beside `attention.key_length` without moving
  `expected` — that was documentation of the gap, not a fix, and stays
  as published.
- **`vectors/v3/deepseek-coder-v2-16b-lite-instruct-q5_K_M.json`**
  (Task 5) — re-pinned to the measured rule: `expected.kv_bytes_per_token
  276480` under R9, with 331,776 (this erratum's figure) and the two
  latent-cache candidates ruled out by the same measurement (31,104 and
  62,208) pinned as `must_not_equal`.

## Fix chain

1. **SPEC.md R9** (this task, Task 4) — "MLA, separate widths":
   `kv_bytes_per_token = attention_layer_count * head_count_kv *
   (key_length + value_length) * bytes_per_element` when
   `attention.value_length` is present and differs from
   `attention.key_length`. R2 defers to R9 in that case (R2 itself
   amended non-silently, dated 2026-08-28, to say so).
2. **`vectors/v3`** (Task 5) — the measured figure enters a new,
   non-frozen vector set; v1/v2 keep their original numbers per
   Versioning.
3. **assay E3** (upstream, out of scope for this repo — this repo
   writes no code against assay) — an erratum row against the affected
   `deepseek-coder-v2` profiles, and a rebuilt matrix, would close the
   "open" status above the way assay's own E2 closed the qwen3.8-27b
   case.
4. **Both consumers** (bloomery, assay itself, and any other GGUF
   Geometry Contract implementation) — Tasks 6–10 of this plan carry R9
   into whatever this repo's own tooling implements; any consumer that
   vendors `vectors/v3/` inherits the corrected figure the same way v2
   consumers inherited R1/R3/R4/R6.

## What closes the "open" status above

Same shape as the qwen3.8-27b erratum's closing condition: an upstream
correction in assay — an erratum row against the affected profiles and
a rebuild of the published matrix. That is assay's call, not this
repo's; this repo measures nothing and derives nothing (design spec
section 0), and its part of the fix chain (R9, `vectors/v3`) is already
closed by this task and Task 5.
