# Remediation Summary: granite_4_0_h_gguf-causal_lm-pytorch-H_TINY_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_4_0_h_gguf/causal_lm/pytorch-H_TINY_GGUF-single_device-inference]

## Result
FAIL — Mamba2 SSM segment_sum gives PCC=0.082 on TT silicon after loader fixes (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
mamba2-ssm-segment-sum-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

After loader fix: E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.08201872643249807. Required: pcc=0.99.

## Root cause

**Two independent bugs:**

**Bug 1 (loader — fixed):** The GGUF file uses architecture name `granitehybrid`, which was not registered in transformers 5.x `GGUF_SUPPORTED_ARCHITECTURES`. The loader raised `ValueError: GGUF model with architecture granitehybrid is not supported yet.` This required a full GGUF loading pipeline patch: architecture registration, tokenizer converter, tensor processor for the hybrid MoE+SSM format, and config remapping.

A second loader bug caused the original INTERNAL error 13: `GraniteMoeHybridTopKGating.forward` calls `expert_size.tolist()` on a device tensor, triggering a PJRT device-to-host transfer that TT silicon does not support. The standard `split(expert_size)` pattern also falls through to a weight gather that would overflow L1 CB budget. Both are fixed by replacing with a static per-expert masked matmul (64-expert unrolled loop).

**Bug 2 (tt-mlir/tt-metal — unfixed, Tier B):** After the loader fixes the test runs to completion but produces PCC=0.082 on TT silicon — essentially uncorrelated with the CPU reference. A CPU-only forward pass of the patched model confirms the MoE dispatch patch is numerically correct (attention-only PCC=1.0, max diff=0.015625). The failure is therefore in how TT silicon executes the Mamba2 SSM layers (35 of 40 layers are Mamba2). The primary suspect is `segment_sum`, a lower-triangular cumsum with `-inf` masking used in the Mamba2 recurrence. This operation involves `torch.cumsum`, `masked_fill(-inf)`, and exponentiation. A similar CumSumOp pathology was observed in KORMo-VL (Tier B). With 35 SSM layers compounding, even a small per-layer error would produce essentially random output.

## Fix

**Bug 1 (committed):** `tt-xla/third_party/tt_forge_models/granite_4_0_h_gguf/causal_lm/pytorch/loader.py` — two commits on the `arch-c-36-tt-xla-dev/nsmith/hf-bringup-5` branch of tt-forge-models:

1. `4c406dc6f4` — register `granitehybrid` GGUF architecture, tensor processor for MoE+SSM weights, config remapping (`granitehybrid` → `granitemoehybrid`), and GGUF field extraction (mamba_n_heads, mamba_d_head, layer_types, num_local_experts, etc.)

2. `d1c1d3cbc4` — replace `expert_size.tolist()` D2H transfer with static per-expert masked matmul

**Bug 2 (proposed fix, not attempted):** The `segment_sum` function in `GraniteMoeHybridMambaLayer.torch_forward` (transformers `modeling_granitemoehybrid.py`) generates a StableHLO subgraph containing cumsum + masked_fill + exp operations. The fix would live in tt-mlir's lowering of `stablehlo.reduce` (cumsum) or in the handling of `-inf` values in masked tensor operations. Debugging requires isolating which op first diverges across the 35 SSM layers on silicon.

## Tier B justification
`cross-cutting` — if the cumsum or masked_fill lowering is wrong, fixing it would affect all Mamba-family models. The exact mechanism within the SSM recurrence is unknown without on-silicon op-level debugging.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    4429.18s (1:13:49)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/granite_4_0_h_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d1c1d3cbc44f8d13da540b9b7cb67536ead409d6 |
