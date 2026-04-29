# Remediation Summary: axk1_2layers-causal_lm-pytorch-axk1_2layers-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[axk1_2layers/causal_lm/pytorch-axk1_2layers-single_device-inference]

## Result
FAIL — model's custom MoE routing calls `.cpu().numpy()` on a device tensor, requiring device-to-host transfer not supported by TT's PJRT infrastructure

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}):
got AttributeError("'ndarray' object has no attribute 'add'")

from user code:
   File "modeling_axk1.py", line 577, in moe_infer
    end_idx = start_idx + num_tokens
  File "tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

Two bugs were encountered in sequence:

**Bug 1 (loader, fixed):** `DynamicCache.get_usable_length` was removed in transformers 5.x. The model's custom `modeling_axk1.py:1420` calls it when `use_cache=True`. Fixed by setting `model.config.use_cache = False` in the loader, and removing `padding="max_length"` (which causes TT attention-mask PCC issues for short inputs).

**Bug 2 (unfixed, Tier B):** The model's custom `MoEMLP.moe_infer` (modeling_axk1.py:533–599) computes expert token routing as:
```python
tokens_per_expert = tokens_per_expert.cpu().numpy()
start_idx = 0
for i, num_tokens in enumerate(tokens_per_expert):
    end_idx = start_idx + num_tokens
    ...
```
The `.cpu().numpy()` converts a device tensor to a numpy array. TT's `TorchFunctionOverride` (`TorchFunctionMode`) intercepts the subsequent `start_idx + num_tokens` arithmetic, capturing numpy scalar operations as tensor FX nodes. TorchDynamo's fake-tensor validation then fails because the FX node tries to call `ndarray.add` on a FakeTensor.

On TT silicon, even if compilation were fixed (e.g., by adding graph breaks), the `.cpu()` call during inference would trigger a device-to-host transfer that TT's PJRT infrastructure does not support, failing with `INTERNAL: Error code: 13`.

## Root cause
The `thkim93/axk1-2layers` model (AXK1 architecture) has 192 routed MoE experts per layer and uses a custom `moe_infer` implementation that explicitly transfers routing tensors from device to CPU (`tokens_per_expert.cpu().numpy()`) to perform Python-loop-based expert dispatch. This pattern requires device-to-host tensor transfer during inference forward, which TT's PJRT bridge (`tt-xla`) does not implement. The compilation failure is a symptom: TT's `TorchFunctionOverride` (a `TorchFunctionMode` context active globally) intercepts numpy+tensor arithmetic from the MoE routing code, causing TorchDynamo to record numpy method calls as tensor FX nodes, then fail when validating those nodes with fake tensors.

## Fix
**Partial loader fix applied** (loader layer, committed):
- `axk1_2layers/causal_lm/pytorch/loader.py`: Set `model.config.use_cache = False` after `from_pretrained` to avoid `DynamicCache.get_usable_length` (removed in transformers 5.x).
- `axk1_2layers/causal_lm/pytorch/loader.py`: Removed `padding="max_length"`, `truncation=True`, `max_length=` from `load_inputs` to prevent TT attention-mask PCC degradation on short inputs.

**Proposed fix for Tier B bug** (not attempted):
Supporting `.cpu()` tensor transfers during inference would require implementing device-to-host tensor materialization in TT's PJRT client (`tt-xla/pjrt_implementation`). This is a cross-cutting infrastructure change — it would affect all models, not just this one.

Alternatively, the model's `moe_infer` could be replaced with a vectorized batched-matmul implementation (no numpy, no `.cpu()` call), but that requires reimplementing the model's MoE routing logic — outside scope for a loader fix.

## Tier B justification
cross-cutting — the `.cpu().numpy()` pattern in MoE routing requires device-to-host transfer infrastructure in the PJRT layer; this affects any model using similar expert-routing patterns and cannot be fixed with a scoped change to one or two files.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b (device initialized; test failed during CPU compilation phase before silicon run)
- Duration: 47.99s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/axk1_2layers/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6f4acb602cef2ceba0b5bd5c0ef6a7f43e5f7b6c |
| tt-forge-models | 0caa5a5fde27faf8388d778c7cfdaa10cf73bd98 |
