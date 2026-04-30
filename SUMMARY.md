# Remediation Summary: huihui_hy_mt1_5_7b_abliterated_i1_gguf-causal_lm-pytorch-HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_hy_mt1_5_7b_abliterated_i1_gguf/causal_lm/pytorch-HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF-single_device-inference]

## Result
FAIL — model loads and runs on TT silicon (pcc=0.9711) but falls below required pcc=0.99; gap is from TT bf16 matmul accumulation (ttmlir-f32-precision-not-preserved, Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (in CI where `gguf` was not installed):
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After adding `gguf>=0.10.0` to requirements.txt, second failure:
```
ValueError: GGUF model with architecture hunyuan-dense is not supported yet.
```

After registering `hunyuan-dense` architecture, remaining failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9710840571037413. Required: pcc=0.99.
```

## Root cause
Three bugs were found: two in the loader and one in the compiler stack.

**Bug 1 (loader):** The model directory had no `requirements.txt`, so the `gguf` package was not installed in CI's isolated environment. Transformers raises `ImportError("Please install torch and gguf>=0.10.0...")` at the earliest GGUF loading call. Fix: add `gguf>=0.10.0` to `requirements.txt`.

**Bug 2 (loader):** The GGUF file's `general.architecture` is `hunyuan-dense`. This architecture was not in transformers 5.x `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, or `GGUF_TO_FAST_CONVERTERS`. Transformers raises `ValueError: GGUF model with architecture hunyuan-dense is not supported yet.` The corresponding transformers model type is `hunyuan_v1_dense` (`HunYuanDenseV1Config`/`HunYuanDenseV1ForCausalLM`). An additional complication: `get_gguf_hf_weights_map` looks up the GGUF architecture name in `gguf.MODEL_ARCH_NAMES` (using the HF `model_type` as the lookup value), but `MODEL_ARCH_NAMES` stores `"hunyuan-dense"` (the GGUF arch name), not `"hunyuan_v1_dense"` (the HF model type). Since the patcher chain from other GGUF loaders drops `model_to_load`, a thread-local approach is used to recover it. Fix: register `hunyuan-dense` in all GGUF tables, use `CONFIG_MAPPING.register` to map the hyphenated arch key to `HunYuanDenseV1Config`, patch `get_gguf_hf_weights_map` to remap `hunyuan_v1_dense` → `hunyuan-dense` when looking up the GGUF tensor name map, and use a `_fixed_load` wrapper + thread-local to thread `model_to_load` through the broken patcher chain.

**Bug 3 (compiler stack):** After both loader fixes the model loads and runs on TT silicon. CPU fp32 vs CPU bf16 PCC is 0.9981 (the weights are quantized Q4_K_M and loaded as bf16 — minimal quantization noise). TT vs CPU produces PCC=0.9711, a ~0.027 gap. This gap accumulates over 32 transformer decoder layers from TT's bf16 matmul accumulation (TT hardware accumulates in bf16; CPU bf16 internally accumulates in fp32). This is the known `ttmlir-f32-precision-not-preserved` Tier B bug — cross-cutting across all matmul lowerings in tt-mlir/tt-metal, not fixable in scope.

## Fix
**Fix 1** — commit `b32c0c6ade` on `remediation/huihui_hy_mt1_5_7b_abliterated_i1_gguf-...` (tt_forge_models):
- `huihui_hy_mt1_5_7b_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt`: added with `gguf>=0.10.0`.

**Fix 2** — commits `ebc46ed1a8`, `03e8317e47`, `9be4b4da36`, `225e0d66f7`, `628eb513ee` on `remediation/huihui_hy_mt1_5_7b_abliterated_i1_gguf-...` (tt_forge_models):
- `huihui_hy_mt1_5_7b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`:
  - Register `hunyuan-dense` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with the standard llama-like field mapping (block_count, embedding_length, feed_forward_length, attention.head_count, attention.key_length, rope.freq_base, etc.).
  - Register `GGUFGPTConverter` for `hunyuan-dense` tokenizer in `GGUF_TO_FAST_CONVERTERS`.
  - Register `LlamaTensorProcessor` for `hunyuan-dense` in `TENSOR_PROCESSORS`.
  - Use `CONFIG_MAPPING.register("hunyuan-dense", HunYuanDenseV1Config)` to map the hyphenated arch name for `AutoConfig.from_pretrained`.
  - Patch `get_gguf_hf_weights_map` to remap `model_type="hunyuan_v1_dense"` → `"hunyuan-dense"` (the gguf-py arch name) before the tensor name map lookup.
  - Install a `_fixed_load` wrapper around `load_gguf_checkpoint` during model loading that captures `model_to_load` in thread-local storage, then calls the chain without it; `_patched_get_gguf_hf_weights_map` recovers the model from thread-local when the chain drops it.

**Fix 3 (proposed, Tier B):** Profile layer-by-layer PCC between TT and CPU bf16 to identify which matmul accumulates the most error. Fix would live in tt-mlir's matmul/attention lowering or tt-metal's bf16 accumulation kernels. Affects all bf16 transformer models on TT hardware.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting — `ttmlir-f32-precision-not-preserved` affects every matmul lowering in tt-mlir and tt-metal. Fixing it requires coordinated changes across multiple files in both repos (MLIR lowering passes and tt-metal BFLOAT16 kernels), far exceeding the one-or-two-file Tier A scope.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    400.49s (0:06:40)
- Tier A attempts: N/A

## Files changed
- `huihui_hy_mt1_5_7b_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `huihui_hy_mt1_5_7b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | de18229371dac9facf3219b1fb5a60b3fa5400f8 |
| tt-forge-models | 628eb513ee6500189a5d98ffe9bd6befe9ebc05b |
