# Remediation Summary: deepseek-deepseek_v3-pytorch-Tiny_Random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3/pytorch-Tiny_Random-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
deepseek-v3-moe-infer-numpy-cpu-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None

Root: `moe_infer` in DeepSeek V3's modeling_deepseek.py calls
`tokens_per_expert.cpu().numpy()` to compute per-expert token counts.
Under Dynamo's TorchFunctionMode / FakeTensorMode (triggered by
`torch.compile(backend="tt")`), FakeTensors cannot be converted to
numpy, raising a TorchRuntimeError during tracing.

## Root cause
The DeepSeek V3 `moe_infer` method (modeling_deepseek.py:535) uses
`tokens_per_expert.cpu().numpy()` to iterate over expert token slots.
This is a host-sync point that Dynamo cannot trace: FakeTensors have no
numpy representation, so the call raises `TorchRuntimeError` inside the
TorchFunctionMode intercept in tt_torch.

The fix belongs in the loader layer: replace `moe_infer` with a
pure-tensor implementation that Dynamo can trace symbolically.

A secondary bug appeared once the numpy call was removed: the routing
scores (`topk_weight`) are float32, so the routing-weighted sum
upcasted the bf16 expert outputs to float32. This float32 tensor
propagated through the residual connection and caused a dtype mismatch
at `lm_head` (bf16 weights vs float32 input). Fixed by casting the
return value to `x.dtype`.

## Fix
**Repo**: tt-forge-models  
**Branch**: remediation/deepseek-deepseek_v3-pytorch-Tiny_Random-single_device-inference  
**File**: `deepseek/deepseek_v3/pytorch/loader.py`

Added `_patch_moe_layers(model)` called after `from_pretrained`, which
monkey-patches `type(module).moe_infer` on every `DeepseekV3MoE` layer.

The replacement `_batched_moe_infer`:
1. Stacks all expert `gate_proj`, `up_proj`, `down_proj` weights into
   `[E, inter, hidden]` tensors (E = 256 for the full model).
2. Runs 3 `torch.bmm` calls (gate, up, down) instead of 256 sequential
   expert module calls, producing O(1) MLIR graph nodes instead of O(256).
3. Computes routing via `scatter_add` on a float32 routing matrix, then
   takes the weighted sum over experts.
4. Casts the result back to `x.dtype` (bf16) to prevent the router's
   float32 scores from upcasting the expert outputs and breaking `lm_head`.

Commits:
- `1eec8a64bf` — Fix DeepSeek-V3 MoE: batched bmm to avoid O(256) MLIR graph
- `c6f00c47a2` — Fix dtype cast in batched moe_infer: router scores are float32, cast back to x.dtype

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    244.35s (0:04:04)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/deepseek/deepseek_v3/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c6f00c47a21a1d06eea980708a255323817edc24 |
