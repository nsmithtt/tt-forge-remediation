# Remediation Summary: llmfan46_qwen3_5_9b_ultra_heretic_i1_gguf-causal_lm-pytorch-9B_ultra_heretic_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llmfan46_qwen3_5_9b_ultra_heretic_i1_gguf/causal_lm/pytorch-9B_ultra_heretic_i1_GGUF-single_device-inference]

## Result
FAIL — Loader fixed (TypeError: model_to_load kwarg in GGUF patch chain); residual PCC 0.8680 vs required 0.99 is a Tier B TT BF16 precision bug in Qwen3.5 SSM linear-attention layers

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-ssm-linear-attn-precision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
After loader fix:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8680141510882697. Required: pcc=0.99.
```

## Root cause
Two bugs:

**Bug 1 (loader, fixed):** The `llmfan46_qwen3_5_9b_ultra_heretic_i1_gguf` loader called `AutoModelForCausalLM.from_pretrained` without registering the `qwen35` GGUF architecture or installing any `load_gguf_checkpoint` wrapper. During pytest collection, other loaders patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with functions that don't accept the `model_to_load` kwarg added in transformers 5.x. Because `modeling_utils.py` does a lazy import (`from .modeling_gguf_pytorch_utils import load_gguf_checkpoint`) at call time, one of those bad wrappers was invoked with `model_to_load=dummy_model`, raising TypeError.

**Bug 2 (tt-mlir, unfixed):** The Qwen3.5-9B model is a hybrid architecture: 24 of 32 decoder layers use SSM-style "linear attention" (`full_attention_interval=4`). On CPU x86, BF16 matmul upscasts to FP32 for accumulation (BLAS standard), giving CPU BF16 PCC = 0.973 vs FP32. On TT hardware, BF16 operations accumulate in BF16 natively; the additional BF16 accumulation error across 24 SSM layers drives TT PCC down to 0.868, a gap of 0.105 beyond what BF16 alone explains.

## Fix
**Bug 1 (loader):** Three commits on the remediation branch of `tt_forge_models` (commit `61a6b7f956`):
1. `163c9d99bc` — register `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, `TENSOR_PROCESSORS`; add `_Qwen35TensorProcessor` for `ssm_conv1d` shape and `dt_bias` name mapping; patch `get_gguf_hf_weights_map` to remap `qwen3_5_text`→`qwen35`.
2. `df9c2aa9eb` — re-install patches at call time inside `_qwen35_gguf_context()` context manager, ensuring they survive late-import overrides by other loaders.
3. `61a6b7f956` — replace import-time patching with BFS over `__globals__` + `__closure__` to find the real `load_gguf_checkpoint` function at call time regardless of import order, then install a `*args, **kwargs` wrapper across all four binding sites.

**Bug 2 (proposed):** Add FP32 accumulation precision to the BF16 lowering of matmul and potentially exp/cumsum ops used inside Qwen3.5 linear attention blocks. This would require changes across multiple lowering passes in tt-mlir (matmul, elementwise exp, scatter/gather for state transitions) — cross-cutting, hence Tier B.

## Tier B justification
`cross-cutting` — Fixing BF16 accumulation precision in SSM linear-attention layers requires changes across multiple lowering passes in tt-mlir (matmul accumulation, elementwise ops, state-space scan). A single-function scoped fix cannot close the 0.105 PCC gap (CPU BF16 = 0.973, TT BF16 = 0.868) across 24 SSM layers, each with multiple interdependent ops.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    3931.08s (1:05:31) after loader fix
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/llmfan46_qwen3_5_9b_ultra_heretic_i1_gguf/causal_lm/pytorch/loader.py` (loader fix on remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 61a6b7f95601223fde0fff814da6db7df21be588 |
