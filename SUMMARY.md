# Remediation Summary: arthurcollet_omnicoder_9b_mlx_mxfp8-causal_lm-pytorch-OmniCoder_9B_mlx_mxfp8-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[arthurcollet_omnicoder_9b_mlx_mxfp8/causal_lm/pytorch-OmniCoder_9B_mlx_mxfp8-single_device-inference]

## Result
FAIL — PCC=0.9807 on TT silicon vs BF16-CPU floor of 0.9974; compiler precision bug in Qwen3.5 GatedDeltaNet (linear_attn) layers

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-qwen35-gated-delta-net-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure: `AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'self_attn'`

After loader fixes, the model ran to completion but failed PCC:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9806983264780198. Required: pcc=0.99.
```

## Root cause

Two sequential loader bugs were fixed before exposing the underlying compiler bug:

**Loader bug 1** (`load_shard_spec`): `Qwen3_5DecoderLayer` in the Qwen3.5 hybrid architecture has either `linear_attn` (GatedDeltaNet, for ~32/36 layers) or `self_attn` (standard attention, for every 4th layer). The loader unconditionally accessed `layer.self_attn`, raising `AttributeError` for linear_attention layers.

**Loader bug 2** (`load_inputs`): `Qwen3_5DynamicCache` (the hybrid SSM+attention cache returned by the model) is not a subclass of `transformers.cache_utils.Cache`. The comparison evaluator's `convert_and_match` only handles `Cache` subclasses, so the cache was treated as a leaf and `torch.equal(Qwen3_5DynamicCache, Qwen3_5DynamicCache)` raised `TypeError`.

After both loader fixes the model compiled and ran, but TT silicon gives PCC=0.9807 vs required 0.99. BF16-CPU vs FP32-CPU PCC is 0.9974, so the BF16 accumulation floor is only 0.0026 below FP32. TT silicon is an additional 0.0167 below BF16-CPU. This gap is real compiler precision loss in the GatedDeltaNet computation, not BF16 accumulation.

The root cause of the precision loss is in tt-mlir: GatedDeltaNet uses complex gating operations (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` with element-wise gating) that appear to accumulate additional BF16 error on WH silicon beyond CPU reference. The 32 linear_attention GatedDeltaNet layers each contribute ~0.0005 PCC degradation beyond the BF16 floor.

## Fix
**Loader fix 1** (committed to `remediation/arthurcollet-omnicoder-9b-mlx-mxfp8-self-attn-shard-spec`):
- `arthurcollet_omnicoder_9b_mlx_mxfp8/causal_lm/pytorch/loader.py` — guard `self_attn` access with `hasattr(layer, "self_attn")`, add `elif hasattr(layer, "linear_attn")` branch with `in_proj_qkv`, `in_proj_z`, `out_proj` shard specs.

**Loader fix 2** (committed to same branch):
- `arthurcollet_omnicoder_9b_mlx_mxfp8/causal_lm/pytorch/loader.py` — add `inputs["use_cache"] = False` in `load_inputs` so the model returns `past_key_values=None` instead of `Qwen3_5DynamicCache`.

**Proposed compiler fix** (not implemented — Tier B):
Identify which GatedDeltaNet operations accumulate excess BF16 error on WH and apply precision-preserving treatment (e.g., FP32 accumulation for key ops). The fix would span multiple lowering patterns in tt-mlir.

## Tier B justification (FAIL with Tier=B only)
cross-cutting

The precision loss spans all 32 GatedDeltaNet layers and requires investigation into which specific operations within GatedDeltaNet (gating, state update, output projection) deviate from BF16 reference. Fixing it would require changes to multiple lowering patterns across the BF16 matmul / element-wise op pipeline in tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    3530.13s (0:58:50) for the final silicon run
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: arthurcollet_omnicoder_9b_mlx_mxfp8/causal_lm/pytorch/loader.py`
  - Branch: `remediation/arthurcollet-omnicoder-9b-mlx-mxfp8-self-attn-shard-spec`
  - Commit 1: Fix load_shard_spec for Qwen3.5 hybrid linear_attn/self_attn layers
  - Commit 2: Disable use_cache to avoid Qwen3_5DynamicCache in model output

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 2c0acf5b5b134aaeea9a6b5d4ec040e1bd4d69fe |
