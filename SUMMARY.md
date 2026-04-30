# Remediation Summary: infinite_talk-pytorch-lightweight_Single_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[infinite_talk/pytorch-lightweight_Single_Q4_K_M-single_device-inference]

## Result
FAIL — loader fix committed; silicon verification blocked by gated HF repo on this machine

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-parameter-torch-function-dynamo-recursion

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded
```

## Root cause
The loader calls `WanTransformer3DModel.from_single_file()` with `GGUFQuantizationConfig`, which
stores all weights as `GGUFParameter` tensor subclasses. When TorchDynamo traces the model,
`GGUFParameter.__torch_function__` calls `super().__torch_function__()`, which under the Dynamo
tracing context enters an infinite recursive loop, exhausting the Python call stack. This is the
same bug as the `flux_1_schnell_gguf` loader (fingerprint: gguf-parameter-torch-function-dynamo-recursion).

## Fix
In `infinite_talk/pytorch/loader.py`, after `WanTransformer3DModel.from_single_file()` returns,
immediately dequantize the weights to plain `nn.Linear` layers before TorchDynamo sees the model:

```python
from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear

_dequantize_gguf_and_restore_linear(self._transformer)
self._transformer.is_quantized = False
self._transformer = self._transformer.to(dtype=compute_dtype)
```

`_dequantize_gguf_and_restore_linear` converts every `GGUFParameter`-backed linear layer to a
plain `nn.Linear` with standard `torch.Tensor` weights. Clearing `is_quantized` allows the
subsequent `.to(dtype=...)` call to proceed without hitting the `ModelMixin.to()` guard that
raises `ValueError: Casting a quantized model to a new dtype is unsupported`.

Committed to: `remediation/infinite_talk-pytorch-lightweight_Single_Q4_K_M-single_device-inference`
in `tenstorrent/tt-forge-models`
Commit: `21f746eb88145ed65418025ec484ebcc8d8d78cd`
File: `infinite_talk/pytorch/loader.py`

## Verification
- pytest exit: not-run
- Hardware:    blackhole-p150b
- Duration:    n/a
- Tier A attempts: N/A

Silicon verification was not possible because `lightweight/InfiniteTalk` is a gated HuggingFace
repository and the HF token on this machine (`bh-lb-13-tt-forge-remediation-7`) does not have
download access approved. The model downloaded successfully in the original CI run on
`bh-lb-12-tt-xla-dev-1` (2026-04-25), confirming that access is token/account-specific. The fix
is structurally identical to the verified `flux_1_schnell_gguf` loader fix for the same bug.

## Files changed
- `infinite_talk/pytorch/loader.py` (in tt-forge-models, remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 21f746eb88145ed65418025ec484ebcc8d8d78cd |
