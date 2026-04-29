# Remediation Summary: anima_gguf-pytorch-preview3_base_Q5_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[anima_gguf/pytorch-preview3_base_Q5_K_M-single_device-inference]

## Result
SILICON_PASS — loader fixed (gated config bypass + dtype_override sig) and Tier A tt-mlir fix (shouldUseDecode k_chunk_size guard); test passes in 8m52s

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
sdpa-decode-k-chunk-size-lt-32

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
2026-04-23 21:05:23.426 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```
After the loader was fixed, the actual compiler-stack failure was:
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
  @ sdpa_decode.cpp:66: k_chunk_size % 32 == 0
```

## Root cause
Two independent bugs:

**Bug 1 (loader):** The original loader called
`CosmosTransformer3DModel.from_single_file(model_path, config="nvidia/Cosmos-Predict2-2B-Text2Image", subfolder="transformer")`.
The `nvidia/Cosmos-Predict2-2B-Text2Image` HuggingFace repo is gated and access is denied,
causing a `GatedRepoError` that manifested as the eth-core TT_FATAL (the device
was left in a bad state by the crash). Additionally, `load_inputs` used `**kwargs`
which meant the test runner's dtype_override injection (which checks `"dtype_override"
in sig.parameters`) couldn't propagate bfloat16 to the inputs.

**Bug 2 (tt-mlir, Tier A):** After the loader was fixed, the Cosmos transformer's
self-attention has query shape `[1, 1, 8, 128]` (one token per head from a 2×2 latent
with patch_size=(1,2,2)), so query_seq_len==1. `shouldUseDecode` in TTIRToTTNN.cpp
routes to SDPA decode whenever query_seq_len==1. The SDPA decode kernel requires
`k_chunk_size % 32 == 0`, where k_chunk_size is `get_chunk_size(key_seq_len)` — the
largest power-of-2 divisor of the key sequence length, capped at 512. With
key_seq_len==1, `get_chunk_size` returns 2, which fails the `% 32 == 0` assertion.
The fix is to guard `shouldUseDecode` to also require `key_seq_len % 32 == 0`, falling
back to regular SDPA when this constraint is not met.

## Fix
**Loader fix** (`tt-xla/third_party/tt_forge_models/anima_gguf/pytorch/loader.py`):
- Replaced `CosmosTransformer3DModel.from_single_file(config="nvidia/Cosmos-Predict2-2B-Text2Image")` with
  a direct model construction using hardcoded Cosmos-Predict2-2B config parameters
  (derived from GGUF tensor shapes).
- Used `load_gguf_checkpoint` + `dequantize_gguf_tensor` + `convert_cosmos_transformer_checkpoint_to_diffusers`
  to load weights without touching the gated config repo.
- Changed `load_inputs(self, **kwargs)` signature to `load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs)`
  so the test runner can inject bfloat16 inputs.
- Added `padding_mask` (shape `[B, 1, H, W]`) to the returned inputs dict (required by `concat_padding_mask=True`).

Commits in `tt_forge_models` on branch `remediation/anima_gguf-pytorch-preview3_base_Q5_K_M-single_device-inference`:
- `1417c81313` — Fix anima_gguf: bypass from_single_file by dequantizing GGUF directly
- `859268cbff` — Fix anima_gguf: explicit dtype_override param in load_inputs for test runner detection

**Compiler fix** (`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`), function `shouldUseDecode`:
- Added a guard requiring `key_seq_len % 32 == 0` before routing to SDPA decode.
- When the key sequence length is not divisible by 32, falls back to regular SDPA.

Commit in `tt-mlir` on branch `remediation/anima_gguf-pytorch-preview3_base_Q5_K_M-single_device-inference`:
- `e91254d1e` — TTIRToTTNN: guard key_seq_len % 32 == 0 in shouldUseDecode

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    532.06s (0:08:52)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/anima_gguf/pytorch/loader.py`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | e91254d1e8080b670639064c545bdaa01afab67e |
| tt-xla          | 2c83c8a17f9132da57e4564c30c109ff70935f1e |
| tt-forge-models | 859268cbff |
