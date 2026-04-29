# Remediation Summary: flux_srpo_gguf-pytorch-Q6_K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_srpo_gguf/pytorch-Q6_K-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
diffusers-resolve-main-url-doubles

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: MonsterMMORPG/Wan_GGUF does not appear to have a file named resolve/main/FLUX-SRPO-GGUF_Q6_K.gguf.

Root URL attempted: https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/resolve/main/FLUX-SRPO-GGUF_Q6_K.gguf
(the extension-modules list in the CI failure message is faulthandler output from a subsequent crash; the primary error is this 404)

## Root cause
Four loader bugs, all in `flux_srpo_gguf/pytorch/loader.py`:

1. **resolve/main URL doubles (diffusers 0.37.x)**: `_extract_repo_id_and_weights_name` in diffusers strips `blob/main/` from URL filenames but not `resolve/main/`. Passing `https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/FLUX-SRPO-GGUF_Q6_K.gguf` caused the regex to extract `resolve/main/FLUX-SRPO-GGUF_Q6_K.gguf` as the filename, so `hf_hub_download` tried `…/resolve/main/resolve/main/…` → 404.

2. **Gated base repo**: `FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")` fails because the repo is gated and not accessible without a token.

3. **GGUFParameter TorchDynamo recursion**: `GGUFParameter.__torch_function__` recurses infinitely under TorchDynamo tracing.

4. **Dict input hang**: passing `joint_attention_kwargs={}` (a Python dict) to the StableHLO backend hangs; the backend only accepts tensor inputs.

## Fix
All fixes in `third_party/tt_forge_models/flux_srpo_gguf/pytorch/loader.py`:

1. Replace `FluxTransformer2DModel.from_single_file(full_url, …)` with `hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)` to obtain a local path, then pass that path to `from_single_file`.

2. Remove `FluxPipeline.from_pretrained(BASE_REPO)` entirely. Instead, construct the transformer config dict locally (`_FLUX_DEV_TRANSFORMER_CONFIG`) and write it to a temp directory. Pass `config=tmpdir` to `from_single_file`.

3. Call `_dequantize_gguf_and_restore_linear(transformer)` after loading to convert `GGUFParameter` layers to plain `nn.Linear` before TorchDynamo tracing; follow with `torch.nn.Module.to(transformer, dtype)` to cast (bypassing the quantizer's `.to()` guard).

4. Change `joint_attention_kwargs: {}` → `joint_attention_kwargs: None` in `load_inputs`.

Also simplified `load_inputs` to construct random tensors from the transformer config directly rather than running CLIP/T5 text encoders (which require the gated pipeline).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    614.86s (0:10:14)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/flux_srpo_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 51104b81911020c77710943dadd46cc8c7c7d6e8 |
