# Remediation Summary: flux_dev_gguf-pytorch-eviation_caesar_Q3_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_dev_gguf/pytorch-eviation_caesar_Q3_K_S-single_device-inference]

## Result
FAIL — Q3_K dequantization uses uint8→float16 bitcast (aten.view.dtype); TT device returns INTERNAL error code 13

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
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

While executing %view_2 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_6, torch.float16), kwargs = {})
Original traceback:
  File ".../diffusers/quantizers/gguf/utils.py", line 380, in dequantize_blocks_Q3_K
    d = d.view(torch.float16).to(dtype)
```

## Root cause
Three loader bugs were fixed:

1. **Missing `gguf>=0.10.0` in requirements.txt**: The original cited ImportError
   (`raise ImportError("Please install torch and gguf>=0.10.0...")`) comes from
   diffusers' GGUF loader when the `gguf` package is absent or too old.

2. **`resolve/main/` URL not stripped by diffusers regex**: The loader passed a
   full `https://huggingface.co/{repo}/resolve/main/{file}` URL to
   `FluxTransformer2DModel.from_single_file`. Diffusers 0.37.1's
   `_extract_repo_id_and_weights_name` regex only strips `blob/main/` (not
   `resolve/main/`), so `weights_name` became
   `resolve/main/experimental-from-f16-caesar/flux1-dev-Q3_K_S.gguf` and the
   HF hub lookup raised `EntryNotFoundError`. Fix: use `hf_hub_download` to
   obtain a local path before calling `from_single_file`.

3. **Gated `black-forest-labs/FLUX.1-dev` config fetch**: After downloading the
   GGUF, `from_single_file` reads GGUF metadata to find
   `pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"`, then calls
   `cls.load_config()` on that gated repo (403 GatedRepoError). Fix: provide an
   inline `_TRANSFORMER_CONFIG` dict written to a temp directory and passed as
   `config=config_dir, subfolder="transformer"` (same pattern as
   `flux_1_fill_dev_gguf`).

After these loader fixes the model loads and compilation begins, but Q3_K
dequantization fails in the compiled graph. `dequantize_blocks_Q3_K` does
`d.view(torch.float16)` — a bitcast from uint8 (1-byte) to float16 (2-byte)
on the TT device. This cross-size dtype view is not supported by the TT
lowering path and produces `INTERNAL: Error code: 13`. This is the same root
cause as the `ttnn-bitcast-cross-size-dtype-unsupported` bug seen in Wan2
Q4_K_M and flux1_arcticlatent Q5_K GGUFs.

## Fix
**Loader fixes (applied)** — `tt_forge_models/flux_dev_gguf/`:
- Created `requirements.txt` with `gguf>=0.10.0`
- `pytorch/loader.py`: imported `hf_hub_download`, added `GGUFParameter.as_tensor`
  patch (DisableTorchFunctionSubclass to prevent __torch_function__ recursion),
  added inline `_TRANSFORMER_CONFIG`, `_make_local_config_dir()` method, changed
  `load_model` to use local GGUF path + local config dir.

**Proposed compiler fix (Tier B, not attempted)**:
Add lowering support for `aten.view.dtype` with cross-size dtype pairs in
`tt-mlir`. Specifically, the uint8 → float16 bitcast (2 uint8 bytes →
1 float16) needs a proper lowering pattern. This would live in the StableHLO
→ TTIR lowering passes, similar in scope to the existing bitcast patterns but
requiring new handling for element-size-mismatch cases.

## Tier B justification
The `aten.view.dtype` cross-size bitcast (uint8 → float16) requires new
lowering infrastructure across multiple passes in tt-mlir. The same class of
bug (`ttnn-bitcast-cross-size-dtype-unsupported`) has been observed in at least
two prior reports (Wan2 Q4_K_M, flux1_arcticlatent Q5_K) without a fix;
fixing it cross-cuts the lowering layer and likely affects multiple ops.
**Tier B indicator**: new-infrastructure

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    188.97s (0:03:08)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/flux_dev_gguf/requirements.txt` (created)
- `tt_forge_models/flux_dev_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e599da2220d4e5f7d3ce0e48e9fd7da3e9e29543 |
| tt-forge-models | 507d167dc51a49748ae32a82c01a2c9318f61016 |
