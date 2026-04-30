# Remediation Summary: flux2_klein_4b_gguf-pytorch-flux2_klein_4b_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux2_klein_4b_gguf/pytorch-flux2_klein_4b_Q4_0-single_device-inference]

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
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

## Root cause
Two loader bugs in `flux2_klein_4b_gguf/pytorch/src/model_utils.py`:

1. **Gated config repo**: `Flux2Transformer2DModel.from_single_file` detects tensor keys matching `CHECKPOINT_KEY_NAMES["flux2"]`, looks up `"flux-2-dev"` in the config mapping, and tries to download `config.json` from `black-forest-labs/FLUX.2-dev` — a gated repo. Without an authorized HF token this raises `GatedRepoError`.

2. **GGUFParameter.__torch_function__ recursion under Dynamo**: Loading with `GGUFQuantizationConfig` wraps linear layers as `GGUFLinear` with `GGUFParameter` weights. Under TorchDynamo, the `GGUFParameter.__torch_function__` dispatch is invoked on every traced op. Each dispatch triggers dequantization logic that calls into the same parameter again, leading to infinite recursion → `RecursionError: maximum recursion depth exceeded`.

## Fix
Three changes in `flux2_klein_4b_gguf/pytorch/`:

1. **`src/transformer_config/config.json`** (new file): Local config derived from GGUF tensor shapes for the 4B architecture (`num_layers=5`, `num_single_layers=20`, `num_attention_heads=24`, `joint_attention_dim=7680`, `guidance_embeds=false`). Passed as `config=_CONFIG_DIR` to `from_single_file` so the gated repo is never contacted.

2. **`src/model_utils.py`**: Added `_dequantize_gguf_and_restore_linear(transformer)` call (diffusers built-in) to replace all `GGUFLinear` → `nn.Linear`, dequantizing Q4_0 weights on CPU. This eliminates `GGUFParameter.__torch_function__` from the call graph entirely. Also added `_dequantize_bf16_params` to fix remaining BF16 `GGUFParameter`s in norm layers, cleared `_hf_quantizer`/`is_quantized` flags, and cast to bfloat16.

3. **`loader.py`**: Removed the `self.transformer.to(dtype_override)` call that fails on quantized (not-yet-dequantized) models. Dtype is now set inside `load_flux2_klein_gguf_transformer`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    235.19s (0:03:55)
- Tier A attempts: N/A

## Files changed
- `flux2_klein_4b_gguf/pytorch/loader.py` — remove `.to(dtype_override)` call on quantized model
- `flux2_klein_4b_gguf/pytorch/src/model_utils.py` — add `_dequantize_gguf_and_restore_linear` + `_dequantize_bf16_params`, clear quantizer flags, cast to bfloat16; add local config path
- `flux2_klein_4b_gguf/pytorch/src/transformer_config/config.json` — new local config for 4B architecture

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | be7d1bada7534fce3b97dfc7e4e139a11f40a8f1 |
| tt-forge-models | 664318c4f2cdb068e0d73d45d04bfeea54a4ff6e |
