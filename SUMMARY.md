# Remediation Summary: baseten_deepseek_v3_fp4-causal_lm-pytorch-V3_FP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[baseten_deepseek_v3_fp4/causal_lm/pytorch-V3_FP4-single_device-inference]

## Result
FAIL — Tier B compiler bug: remote model moe_infer uses .cpu().numpy() for-loop expert dispatch that breaks TorchDynamo tracing during TT device compilation

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-moe-forloop-numpy

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):

```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'. Did you mean: 'get_seq_length'?
```

File: `modeling_deepseek.py:1427` in `DeepSeekV3Model.forward`:
```python
past_key_values_length = past_key_values.get_usable_length(seq_length)
```

Also at lines 797 and 928 in attention forward.

After loader fix, second failure during TT device compilation:

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}):
got AttributeError("'ndarray' object has no attribute 'add'")

from user code:
   File "modeling_deepseek.py", line 579, in moe_infer
    end_idx = start_idx + num_tokens
```

## Root cause

Two bugs found: one loader (fixed) and one compiler-stack (Tier B, unfixed).

**Bug 1 (loader, fixed):** The remote model file `modeling_deepseek.py` from `baseten/DeepSeek-V3-FP4` was written against transformers <5.x API. `DynamicCache.get_usable_length()` was removed in transformers 5.x; the method is called in three places in the model's attention and main forward. The fix restores it as a compatibility shim on `DynamicCache` at module load time.

**Bug 2 (tt-xla, Tier B):** The model's `moe_infer` method (lines 534-600 of `modeling_deepseek.py`) performs expert dispatch via a Python for-loop that first converts expert token counts to a numpy array via `.cpu().numpy()`, then iterates to call each expert individually. During TorchDynamo tracing (used by the TT XLA compiler frontend), `tokens_per_expert.cpu().numpy()` triggers a device-to-host transfer and yields numpy scalars as loop variables. When `start_idx + num_tokens` executes with a numpy scalar `num_tokens` inside the `TorchFunctionMode` context (`tt_torch/torch_overrides.py`), Dynamo wraps it as a FakeTensor operation and fails with `AttributeError("'ndarray' object has no attribute 'add'")`.

This is the same class of bug as Qwen3MoE and Jamba MoE for-loop dispatch failures. Unlike Qwen3MoE (which supports `_experts_implementation = "batched_mm"` as a config switch), this remote model has no alternative implementation path.

## Fix

**Applied (loader):** Added `DynamicCache.get_usable_length` compatibility shim to the loader. The shim matches the removed API's semantics: returns `get_seq_length(layer_idx)` for unbounded caches, or `max_cache_len - new_seq_length` when a static max cache is set and would be exceeded.

File: `baseten_deepseek_v3_fp4/causal_lm/pytorch/loader.py` (tt-forge-models)
Branch: `remediation/baseten_deepseek_v3_fp4-causal_lm-pytorch-V3_FP4-single_device-inference`

**Not applied (compiler-stack, Tier B):** The `moe_infer` for-loop/numpy dispatch requires new infrastructure in tt-xla to either trace through device-to-host transfers with dynamic control flow, or transform for-loop MoE dispatch into batched matmul. The remote model has no `_experts_implementation` config switch unlike Qwen3MoE.

## Tier B justification

**Indicator: new-infrastructure**

The `moe_infer` function contains a Python for-loop over a numpy array derived from a CPU-transferred tensor (`tokens_per_expert.cpu().numpy()`). TorchDynamo cannot trace through this pattern. Fixing it requires either: (a) implementing support for graph breaks around device-to-host transfers in the TT XLA tracing path, or (b) providing a batched-matmul MoE dispatch path in the compiler. Neither is a scoped single-function fix.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 164.73s (second run with loader fix applied)
- Tier A attempts: N/A

## Files changed
- `baseten_deepseek_v3_fp4/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5c43365e8699186d390268606401232a4c1a195a |
