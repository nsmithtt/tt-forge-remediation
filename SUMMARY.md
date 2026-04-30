# Remediation Summary: flux1_arcticlatent_pytorch-dev_Q5_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux1_arcticlatent/pytorch-dev_Q5_K_S-single_device-inference]

## Result
FAIL — Q5_K dequantization uses aten.view.dtype (uint8→float16 bitcast) which ttnn::bitcast does not support for cross-size dtype pairs

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
ttnn-bitcast-cross-size-dtype-unsupported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure (gguf not installed):
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After loader fix 1 (gated HF repo):
```
OSError: black-forest-labs/FLUX.1-dev is not a local folder and is not a valid model identifier
```

After loader fix 2 (inline config):
```
torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded
  File "diffusers/quantizers/gguf/utils.py", line 545, in as_tensor
    return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)
  [repeated 158 times in __torch_function__]
```

After loader fix 3 (as_tensor patch):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
While executing %view_3 : call_function[target=torch.ops.aten.view.dtype](args = (%slice_2, torch.float16), kwargs = {})
  File "diffusers/quantizers/gguf/utils.py", line 336, in dequantize_blocks_Q5_K
    dmin = dmin.view(torch.float16).to(dtype)
```

## Root cause
Three loader bugs masked the terminal compiler issue:

1. **Missing gguf requirement**: `flux1_arcticlatent/pytorch/requirements.txt` did not exist, so `gguf>=0.10.0` was not installed. Diffusers' `load_gguf_checkpoint()` in `model_loading_utils.py:664` raises `ImportError` when `is_gguf_available()` returns False.

2. **Gated config repo**: `FluxTransformer2DModel.from_single_file()` without an explicit config tries to fetch the architecture config from the gated `black-forest-labs/FLUX.1-dev` repository, which requires HuggingFace authentication and license acceptance.

3. **GGUFParameter.as_tensor() infinite recursion**: `GGUFParameter.__torch_function__` intercepts `torch.Tensor._make_subclass` calls because `self` (a `GGUFParameter`) is passed as the data argument, causing infinite recursion. The fix wraps the `_make_subclass` call in `torch._C.DisableTorchFunctionSubclass()`.

After all three loader fixes, the terminal error is a **compiler bug**: Q5_K GGUF dequantization in `dequantize_blocks_Q5_K()` calls `dmin.view(torch.float16)` where `dmin` is a uint8 slice. This maps to `aten.view.dtype` which the TT backend lowers to `ttnn::bitcast`. `ttnn::bitcast` only supports same-size dtype pairs (e.g., bfloat16↔uint16); it cannot bitcast uint8→float16 (1-byte→2-byte).

## Fix
Three loader fixes applied in `tt_forge_models` on branch `remediation/flux1_arcticlatent_pytorch-dev_Q5_K_S-single_device-inference`:

- `flux1_arcticlatent/pytorch/requirements.txt` (new file): `gguf>=0.10.0`
- `flux1_arcticlatent/pytorch/loader.py`:
  - Added `_TRANSFORMER_CONFIG` dict with the standard FLUX.1-dev transformer architecture
  - Added `_make_local_config_dir()` method that writes `config.json` to a temp dir
  - `load_model()` now passes `config=config_dir, subfolder="transformer"` to `from_single_file()`
  - Monkey-patched `GGUFParameter.as_tensor` to use `torch._C.DisableTorchFunctionSubclass()` context manager, breaking the `__torch_function__` recursion loop

Terminal compiler bug (unfixed): `ttnn::bitcast` does not support cross-size dtype pairs. The proposed fix would be to add a dequant/repack kernel that handles the uint8→float16 memory reinterpretation natively, or add a fallback in the `aten.view.dtype` lowering that performs a shift/extract instead of a direct bitcast when sizes differ. This would touch `tt-xla` (lowering) and potentially `tt-mlir` / `tt-metal` (new kernel).

## Tier B justification
Which indicator applies: **new-infrastructure**

The `ttnn::bitcast` operation only supports same-element-size type pairs. Q5_K dequantization requires reinterpreting a byte buffer (uint8) as float16, which is a cross-size bitcast. Supporting this requires either (a) a new TTNN kernel that performs the unpack-and-reinterpret in a single pass, or (b) new lowering infrastructure in tt-xla to decompose `aten.view.dtype` for cross-size pairs into a sequence of shift/mask/cast ops. Neither fits in a single scoped fix.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 282.54s (0:04:42)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/flux1_arcticlatent/pytorch/loader.py` (modified)
- `tt_forge_models/flux1_arcticlatent/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 32d0b9046e420b97ac9f471393c9ea2abdc52c3c |
| tt-forge-models | 12a8d3a20ae56cb9a01366a5e15b7d256c4e29c2 |
