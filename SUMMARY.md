# Remediation Summary: calcuis_wan2_gguf-pytorch-2.2_I2V_LowNoise_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[calcuis_wan2_gguf/pytorch-2.2_I2V_LowNoise_Q4_K_M-single_device-inference]

## Result
FAIL — GGUFParameter.__torch_function__ recurses infinitely under torch._dynamo during GGUF weight dequantization

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
gguf-ggufparameter-torch-function-dynamo-recursion

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
from user code:
   File ".../diffusers/quantizers/gguf/utils.py", line 520, in dequantize_gguf_tensor
    tensor = tensor.as_tensor()
  File ".../diffusers/quantizers/gguf/utils.py", line 545, in as_tensor
    return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)
  File ".../diffusers/quantizers/gguf/utils.py", line 564, in __torch_function__
    result = super().__torch_function__(func, types, args, kwargs)
  [Previous line repeated 158 more times]
```

## Root cause

Two bugs compound here, one fixable (loader) and one not (dynamo bridge).

**Loader bug (fixed):** The original loader passed a bare
`https://huggingface.co/{GGUF_REPO}/{gguf_file}` URL to
`WanTransformer3DModel.from_single_file`. This is not a valid HuggingFace
download URL, and also the loader did not pass any config for I2V variants.
`from_single_file` calls `fetch_diffusers_config` which maps the Wan I2V GGUF
tensor keys to `"wan-i2v-14B"` → fetches `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`.
That config includes `added_kv_proj_dim: 5120` and `image_dim: 1280`, which
create 216 parameters (`attn2.add_k_proj`, `attn2.add_v_proj`,
`attn2.norm_added_k` for all 40 blocks, plus `condition_embedder.image_embedder.*`)
absent from the Wan 2.2 GGUF. With `low_cpu_mem_usage=True` (default when
accelerate is installed), these 216 parameters remain as meta tensors, and
`dispatch_model(model, device_map={"": cpu})` raises
`NotImplementedError: Cannot copy out of meta tensor`.

The fix loads the Wan 2.1 I2V config explicitly, strips the two offending
keys, writes the modified config to a temp directory, and passes that to
`from_single_file`. Also switches to `hf_hub_download` for correct local
caching.

**Compiler-stack bug (Tier B, unfixed):** After the loader fix, the model
loads correctly, but dynamo fails with a `RecursionError` when tracing
`GGUFLinear.forward`. During tracing, `dequantize_gguf_tensor(self.weight)`
calls `tensor.as_tensor()` → `torch.Tensor._make_subclass(torch.Tensor, self,
self.requires_grad)`. Under dynamo, `_make_subclass` is dispatched through
`__torch_function__` because `self` is a `GGUFParameter` subclass. dynamo
intercepts the call before `DisableTorchFunctionSubclass()` can suppress it.
`GGUFParameter.__torch_function__` calls `super().__torch_function__` →
`torch.nn.Parameter.__torch_function__` → `func(*args, **kwargs)` inside
`DisableTorchFunctionSubclass()` (which dynamo ignores) → wraps result with
`GGUFParameter(ret, ...)` → `__torch_function__` fires again → infinite
recursion at depth ~158.

## Fix

**Loader fix (committed):** Strip `added_kv_proj_dim` and `image_dim` from the
Wan 2.1 I2V config before loading the Wan 2.2 GGUF. Use `hf_hub_download`
instead of a bare URL.

File changed: `calcuis_wan2_gguf/pytorch/loader.py` in
`tenstorrent/tt-forge-models` on branch
`remediation/calcuis_wan2_gguf-pytorch-2.2_I2V_LowNoise_Q4_K_M-single_device-inference`
(commit `3d14a32b05`).

**Proposed compiler-stack fix (not attempted):** The `GGUFParameter.__torch_function__`
recursion requires changes in the dynamo bridge to correctly honor
`DisableTorchFunctionSubclass()` when tracing through `torch.nn.Parameter`
subclass `__torch_function__` implementations. This requires either:
(a) Changes in diffusers to make `GGUFParameter.as_tensor()` avoid calling
`_make_subclass` on itself and instead operate on the underlying raw tensor
(obtaining it via `torch.Tensor.data.__get__(self)` or similar without going
through `__torch_function__`), OR (b) Changes in torch-xla's dynamo tracing
to suppress `__torch_function__` dispatch on `_make_subclass` when inside a
`DisableTorchFunctionSubclass` context. Both paths require cross-cutting
changes in diffusers or the dynamo bridge.

## Tier B justification

**cross-cutting**: Fixing the recursion requires either modifying diffusers'
`GGUFParameter` class (third-party library, version-pinned) to avoid
re-entrant `__torch_function__` dispatch, or modifying the dynamo bridge
`DisableTorchFunctionSubclass` context handling — changes that span multiple
libraries and are not scoped to a single function.

Additionally, pre-dequantizing all GGUF weights to bfloat16 as a workaround
would require approximately 28 GB of DRAM (14B params × 2 bytes), exceeding
the 24 GB available on the test device. Using `@torch.compiler.disable` on
`GGUFLinear` to bypass dynamo tracing is a forbidden workaround (CPU offload
of model components).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    267.13s (0:04:27) — test ran to completion, failed with RecursionError
- Tier A attempts: N/A

## Files changed
- `calcuis_wan2_gguf/pytorch/loader.py` (in `tenstorrent/tt-forge-models`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1e331aee56a3163447a2bf478ab5f69806b872dd |
| tt-forge-models | 3d14a32b058621b422e7c20917dc656ca408589f |
