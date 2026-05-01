# Remediation Summary: flux_1_fill_dev_gguf-pytorch-Q5_1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_1_fill_dev_gguf/pytorch-Q5_1-single_device-inference]

## Result
SILICON_PASS

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
E   OSError: YarvixPA/FLUX.1-Fill-dev-GGUF does not appear to have a file named resolve/main/flux1-fill-dev-Q5_1.gguf.

(The task failure message `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` occurs in environments without gguf installed; in the local venv where gguf is installed, the first observable failure is the 404 from the doubled resolve/main path.)

## Root cause
Three bugs in the original loader, all in the loader layer:

1. **URL double resolve/main**: The loader passed `https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/resolve/main/flux1-fill-dev-Q5_1.gguf` to `from_single_file`. diffusers 0.37.1 only strips `blob/main` from HF URLs; a `resolve/main` path is treated as a filename relative to the repo root, so the actual request becomes `.../resolve/main/resolve/main/flux1-fill-dev-Q5_1.gguf` → 404.

2. **Gated config repo**: When `from_single_file` receives a local GGUF path (after fixing bug 1), it inspects the GGUF metadata `general.architecture = "flux"` together with `in_channels=384` to infer `black-forest-labs/FLUX.1-Fill-dev` as the config repo. That repo is gated and returns 403 without a granted token.

3. **GGUFParameter recursion under TorchDynamo**: `GGUFQuantizationConfig` leaves all linear weights as `GGUFParameter` tensor subclass instances. `GGUFParameter.__torch_function__` calls `super().__torch_function__()`, which under TorchDynamo tracing recurses infinitely → `RecursionError`.

## Fix
All three fixes are in `flux_1_fill_dev_gguf/pytorch/loader.py` in tt-forge-models (commit `73988ba741`):

1. Replace the URL string with `hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)` to get a local path — same pattern used by `anima_gguf` and `flux_schnell_gguf` loaders.

2. Materialise a local `transformer/config.json` in a temp dir with the hardcoded FLUX Fill architecture (`in_channels=384, out_channels=64`) and pass `config=config_dir, subfolder="transformer"` to bypass the metadata lookup entirely.

3. After loading, call `_dequantize_gguf_and_restore_linear(transformer)`, then clear `transformer.is_quantized = False`, then `transformer.to(dtype)` so all parameters are plain BF16 tensors before compilation.

Also replaced the pipeline-based `load_inputs` (which required the gated `black-forest-labs/FLUX.1-Fill-dev` tokenizers/VAE) with synthetic random embeddings of the correct shapes.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    890.06s (0:14:50)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: flux_1_fill_dev_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ef94c7b344225438a84c0886346a676a9ba70442 |
| tt-forge-models | 73988ba741159286cbd3614a01a043885c3b7ac6 |
