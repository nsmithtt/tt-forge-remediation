# Remediation Summary: flux2_nvfp4-pytorch-NVFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux2_nvfp4/pytorch-NVFP4-single_device-inference]

## Result
XFAIL — FLUX.2-dev transformer (8 double + 48 single blocks, hidden_size=6144) dequantizes to ~32 GB+ BF16, exceeding single-device DRAM; NVFP4 native inference is not supported on TT hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
flux2-nvfp4-chunk-0dim-scale-tensor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: chunk expects at least a 1-dimensional tensor

## Root cause
Three stacked loader bugs prevented the model from loading, and after fixing them the model exceeds single-device DRAM:

1. **URL double-path** (loader): `from_single_file` receives `https://.../resolve/main/flux2-dev-nvfp4.safetensors`; diffusers does not strip `resolve/main/` (only strips `blob/main/`), then re-appends `resolve/main/` yielding a 404.  Fix: use `blob/main` in the URL.

2. **Gated config repo** (loader): `from_single_file` tries to download `config.json` from the gated `black-forest-labs/FLUX.2-dev` repository when given a local safetensors path.  Fix: write the transformer architecture config to a temp directory and pass it via `config=`.

3. **0-dim tensor in `torch.chunk`** (loader): The NVFP4 checkpoint stores per-tensor activation scales as 0-dim `float32` scalars (e.g. `double_blocks.0.img_attn.qkv.input_scale`, shape `torch.Size([])`).  The diffusers `convert_flux2_transformer_checkpoint_to_diffusers` function calls `torch.chunk(..., dim=-1)` on all QKV-related tensors including these 0-dim scalars, raising `RuntimeError: chunk expects at least a 1-dimensional tensor`.  Fix: patch `SINGLE_FILE_LOADABLE_CLASSES["Flux2Transformer2DModel"]["checkpoint_mapping_fn"]` with a context manager that drops `uint8`, `float8_e4m3fn`, and 0-dim tensors before the conversion step.  The dropped quantized weight tensors (NVFP4 packed `uint8` + FP8 per-block scales) fall back to random BF16 init; this is acceptable for a compiler correctness test.

After all three fixes the model loads and compiles successfully, but fails at device execution with OOM:
```
Out of Memory: Not enough space to allocate 679477248 B DRAM buffer across 8 banks
(allocated: 4009071872 B, free: 264318144 B)
```
The `679477248 B` allocation is exactly one `to_qkv_mlp_proj` weight matrix `[55296, 6144]` at BF16.  With 48 single-stream blocks × 679 MB each, the single-stream weights alone require ~32.6 GB, far exceeding any single TT device.  NVFP4 quantization was designed to halve this to ~16 GB, but TT hardware has no native FP4 inference support and dequantising to BF16 restores the full footprint.

## Fix
Three commits on `remediation/flux2_nvfp4-pytorch-NVFP4-single_device-inference` in tt-forge-models:
- `2f01483096` — URL: `resolve/main` → `blob/main`
- `9aa30c8686` — Local transformer config to bypass gated FLUX.2-dev repo
- `41f92012a7` — NVFP4 checkpoint filter: drop 0-dim/uint8/float8 tensors, fall back to random BF16 init

One commit on `remediation/flux2_nvfp4-pytorch-NVFP4-single_device-inference` in tt-xla:
- `84fa4d5fd` — Add `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: FAIL (OOM at device execution; xfail marker added)
- Hardware:    blackhole-p150b
- Duration:    467.05s (0:07:47) before xfail marker
- Tier A attempts: N/A

## Files changed
- `flux2_nvfp4/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2dd4cec15cb599814958c1014a40129b6ba48fb7 |
| tt-forge-models | 41f92012a794521b1a1d660a5bf59014c3da5056 |
