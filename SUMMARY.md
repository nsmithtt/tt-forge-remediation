# Remediation Summary: flux_schnell_gguf-pytorch-flux1_schnell_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_schnell_gguf/pytorch-flux1_schnell_Q4_0-single_device-inference]

## Result
SILICON_PASS — fixed two loader bugs: gated-repo access via local config.json and GGUFParameter.__torch_function__ infinite recursion fixed by dequantizing before TorchDynamo tracing

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
E   OSError: black-forest-labs/FLUX.1-schnell is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo with `token` or log in with `hf auth login`.
```
(The originally reported RecursionError would surface after fixing the gated-repo issue. Both bugs live in the same loader and are fixed together.)

## Root cause
Two bugs in `flux_schnell_gguf/pytorch/loader.py` and `flux_schnell_gguf/pytorch/src/model_utils.py`:

1. `load_flux_gguf_pipe` called `FluxTransformer2DModel.from_single_file` without a local config, which caused diffusers to read architecture parameters from the gated `black-forest-labs/FLUX.1-schnell` HuggingFace repo. Without auth credentials, this raises OSError.

2. `FluxTransformer2DModel.from_single_file` with `GGUFQuantizationConfig` wraps all weight tensors as `GGUFParameter` instances. When TorchDynamo traces through the model's forward pass, `GGUFParameter.__torch_function__` is called on tensor operations. Its implementation calls `super().__torch_function__()` which wraps results back as `GGUFParameter`, triggering `__torch_function__` again — causing infinite recursion (RecursionError: maximum recursion depth exceeded).

3. The loader also built a full `FluxPipeline` from the gated repo to construct text encoder inputs, creating an unnecessary dependency on a gated model.

## Fix
File: `tt_forge_models/flux_schnell_gguf/pytorch/loader.py`

Three changes (loader.py rewritten, model_utils.py no longer used):
1. **Eliminated FluxPipeline dependency**: Replaced `load_flux_gguf_pipe` (which called `FluxPipeline.from_pretrained(BASE_MODEL, ...)`) with a direct `FluxTransformer2DModel.from_single_file` call using a locally materialised `config.json` containing the known FLUX.1-schnell transformer architecture (`guidance_embeds=False`, `num_layers=19`, `num_single_layers=38`, etc.). This avoids gated repo access entirely.

2. **Dequantized GGUF weights before compilation**: After `from_single_file`, called `_dequantize_gguf_and_restore_linear(self._transformer)` from `diffusers.quantizers.gguf.utils`, then cleared `self._transformer.is_quantized = False`, then cast to `dtype`. This converts all `GGUFParameter` tensors to plain `torch.Tensor` in `nn.Linear` layers, preventing the `__torch_function__` recursion under TorchDynamo.

3. **Replaced real-encoder inputs with synthetic tensors**: `load_inputs` now constructs random tensors for `encoder_hidden_states`, `pooled_projections`, and `txt_ids` directly from `config.joint_attention_dim` and `config.pooled_projection_dim`, without needing any tokenizer or text encoder.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    573.33s (0:09:33)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/flux_schnell_gguf/pytorch/loader.py` (rewritten)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0339ca69961fa3d39f7b39ec3cda0a5f41fd5bbe |
| tt-forge-models | 5e2afb50eb7059350fee787aa9f51302027cd6d8 |
