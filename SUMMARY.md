# Remediation Summary: flux_1_schnell_gguf-pytorch-Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_1_schnell_gguf/pytorch-Q8_0-single_device-inference]

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
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded
...
diffusers/quantizers/gguf/utils.py, line 564, in __torch_function__
    result = super().__torch_function__(func, types, args, kwargs)
[Previous line repeated 158 more times]
```

## Root cause
Two bugs in `flux_1_schnell_gguf/pytorch/loader.py`:

1. The loader built a full `FluxPipeline` from `black-forest-labs/FLUX.1-schnell` (a gated HuggingFace repo) to get tokenizers and text encoders for input construction. This fails locally without authentication and unnecessarily couples the loader to a gated model. The underlying text encoder outputs are not needed for silicon testing — synthetic random tensors suffice.

2. `FluxTransformer2DModel.from_single_file` loads a GGUF file, which wraps all weight tensors as `GGUFParameter` instances. When TorchDynamo traces through the model's forward pass, `GGUFParameter.__torch_function__` is called on tensor operations. Its implementation calls `super().__torch_function__()` which wraps results back as `GGUFParameter`, triggering `__torch_function__` again — causing infinite recursion. The model must be dequantized to plain `nn.Linear` layers before compilation.

Secondary complication: `from_single_file` without a local config attempts to read architecture parameters from the gated `black-forest-labs/FLUX.1-schnell` repo. Providing a local `config.json` with the known FLUX.1-schnell transformer architecture avoids this gated access.

## Fix
File: `tt_forge_models/flux_1_schnell_gguf/pytorch/loader.py`

Three changes:
1. **Eliminated FluxPipeline dependency**: Replaced `FluxPipeline.from_pretrained(BASE_REPO, ...)` with a direct `FluxTransformer2DModel.from_single_file` call using a locally materialized `config.json` containing the known FLUX.1-schnell transformer architecture (`guidance_embeds=False`, `num_layers=19`, `num_single_layers=38`, etc.). This avoids gated repo access entirely.

2. **Dequantized GGUF weights before compilation**: After `from_single_file`, called `_dequantize_gguf_and_restore_linear(self._transformer)` from `diffusers.quantizers.gguf.utils`, then cleared `self._transformer.is_quantized = False`, then cast to `dtype`. This converts all `GGUFParameter` tensors to plain `torch.Tensor` in `nn.Linear` layers, preventing the `__torch_function__` recursion under TorchDynamo.

3. **Replaced real-encoder inputs with synthetic tensors**: `load_inputs` now constructs random tensors for `encoder_hidden_states`, `pooled_projections`, and `txt_ids` directly from `config.joint_attention_dim` and `config.pooled_projection_dim`, without needing any tokenizer or text encoder.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    541.01s (0:09:01)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/flux_1_schnell_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 244a3aab19051aa6ec5244449fc0830a93a9e6de |
| tt-forge-models | 8180191dfad94c572425122cba6bbd1bdde1cb1a |
