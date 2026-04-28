# Remediation Summary: deepseek-deepseek_v3_2_4bit_mlx-pytorch-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_2_4bit_mlx/pytorch-single_device-inference]

## Result
FAIL — tt-mlir lowers 3D expert gather (stablehlo.gather on 256×4096×1024 weight) to ttnn.embedding with 8 MB weight-page CB, overflowing 1.5 MB L1

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-3d-gather-large-page-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Underlying fatal (from device log):
TT_FATAL @ program.cpp:1136: Statically allocated circular buffers grow to 4305920 B which is beyond max L1 size of 1572864 B

## Root cause
Two separate issues were found and resolved in sequence:

**Issue 1 (loader — fixed):** `DeepseekV3NaiveMoe.forward()` used
`.nonzero()` to find active experts, which creates a data-dependent tensor
whose shape is unknown at trace time. This produced a graph break and its
continuation failed to compile on TT, surfacing as `INTERNAL Error code: 13`.
Fix: set `config._experts_implementation = "batched_mm"` in the loader so
the static-shape `batched_mm_experts_forward` path (from
`transformers.integrations.moe`) is selected instead.

**Issue 2 (tt-mlir — unfixed, Tier B):** The `batched_mm` path performs
expert lookup via advanced indexing: `gate_up_proj[expert_ids_clamped]` where
`gate_up_proj` has shape `[num_experts, 2*intermediate, hidden]` =
`[256, 4096, 1024]` (with test-time dims). tt-mlir lowers this
`stablehlo.gather` by:
1. Reshaping the 3D weight to `[256, 4194304]` (flatten last two dims).
2. Calling `ttnn.embedding` with `indices [10, 1]` against the `[256, 4194304]`
   weight table.

The embedding CB allocates one row of the weight table as its page:
`4194304 elements × 2 bytes = 8 MB`, which is 5× the 1.5 MB L1 maximum.
This causes the fatal TT_FATAL at program.cpp:1136 and surfaces as
`INTERNAL Error code: 13`.

The same overflow occurs on the down_proj gather
(`[256, 1024, 2048]` → `[256, 2097152]` → 4 MB page).

## Fix
**Applied (loader):** Set `config._experts_implementation = "batched_mm"` in
`tt-forge-models/deepseek/deepseek_v3_2_4bit_mlx/pytorch/loader.py`, before
`AutoModelForCausalLM.from_config()`. This selects the static-shape expert
dispatch from `transformers.integrations.moe.batched_mm_experts_forward`,
eliminating the `.nonzero()` dynamic shape that caused Error 13 during
compilation (same pattern as the Qwen3MoE fix).

**Not applied (tt-mlir):** The 3D gather lowering in tt-mlir needs to avoid
flattening the weight tensor before passing it to `ttnn.embedding` — or to
use a different lowering (e.g., `ttnn.gather`) that operates on the 3D tensor
natively without materializing an 8 MB CB page. This requires either new
infrastructure in tt-mlir's gather lowering pass or cross-cutting changes to
how large embedding tables are handled, making it Tier B.

## Tier B justification
**new-infrastructure | cross-cutting**

The existing `stablehlo.gather` lowering in tt-mlir dispatches to
`ttnn.embedding` by flattening the last N-1 dims of the weight table. Fixing
this for 3D+ source tensors requires either:
- Teaching the gather lowering to choose a different TTNN primitive (no
  direct equivalent today), or
- Splitting the gather along the batch/sequence dimension and interleaving
  with the embedding lookup.

Either change affects the gather lowering pass and all ops that flow through
it, constituting cross-cutting / new-infrastructure work.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    not-run (test exits before inference with TT_FATAL L1 overflow)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/deepseek/deepseek_v3_2_4bit_mlx/pytorch/loader.py` — added `config._experts_implementation = "batched_mm"`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | efc909ef1c0f34e647814231e5974897bf6ba7a5 |
| tt-forge-models | e094127f06efceb62b160632cc912ebaa58d2c90 |
