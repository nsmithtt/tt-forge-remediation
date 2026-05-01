# Remediation Summary: deepseek-deepseek_v3_1_base-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_1_base/pytorch-single_device-inference]

## Result
FAIL — PCC=0.8123 (required 0.99): BF16 MoE routing sensitivity with random weights causes different expert selection on TT vs CPU

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-moe-routing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure: `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'. Did you mean: 'get_seq_length'?`

After loader fixes applied: `AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8123167912277929. Required: pcc=0.99.`

## Root cause
Two bugs combined.

**Loader bug (fixed):** The remote `modeling_deepseek.py` for `unsloth/DeepSeek-V3.1-Base-BF16` was written for older transformers. In transformers 5.x: `DynamicCache.get_usable_length` was replaced by `get_seq_length`; `DynamicCache.from_legacy_cache` was removed; `is_torch_fx_available` was removed from `transformers.utils.import_utils`. Also, `moe_infer` iterates over `tokens_per_expert.cpu().numpy()` — under Dynamo tracing, numpy integer arithmetic on FakeTensors raises `AttributeError: 'ndarray' object has no attribute 'add'`.

**Compiler bug (Tier B):** After loader fixes, the CPU run passes but TT execution yields PCC=0.81. The `noaux_tc` MoE routing performs two `topk` operations (group selection then expert selection) on gating scores. With random-initialized weights, all expert scores are near-uniform (~1/256). Any small difference in BF16 matmul accumulation order between TT hardware and CPU reference flips group and expert selection. Since different experts have completely independent random weights, routing to a different set of experts produces dramatically different outputs. This PCC is intrinsic to BF16 MoE routing sensitivity under random weights, not a bug in any single lowering function.

## Fix
**Loader fixes** in `deepseek/deepseek_v3_1_base/pytorch/loader.py` (tt-forge-models):
1. Patch `transformers.utils.import_utils.is_torch_fx_available` at module import (missing in transformers 5.x)
2. Patch `DynamicCache.from_legacy_cache` (removed in transformers 5.x)
3. Patch `DynamicCache.get_usable_length` → delegates to `get_seq_length(layer_idx)` (removed in transformers 5.x)
4. After `from_config`, monkey-patch `DeepseekV3MoE.moe_infer` to replace `tokens_per_expert.cpu().numpy()` with `.cpu().tolist()` so Dynamo sees Python ints (not numpy integer objects wrapping FakeTensors)

**Proposed compiler fix (not attempted — Tier B):** Preserve FP32 precision through MoE gating computation (the `topk` scoring matmul and softmax) to match CPU reference regardless of BF16 accumulation order. This is a cross-cutting change across the StableHLO→TTIR lowering passes.

## Tier B justification
Indicator: **cross-cutting**. Fixing MoE routing precision requires preserving FP32 (or bit-exact BF16) accumulation through the gating matmul and topk operations across multiple lowering passes in tt-mlir. It cannot be scoped to a single function or file.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    266.11s (0:04:26)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_1_base/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e3a0e9edc4c83b96fcbd75b65be1580ad8ea27fb |
| tt-forge-models | c94c7439206bbd9271be0836540e5039833701af |
