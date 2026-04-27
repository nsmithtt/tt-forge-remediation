# Remediation Summary: arctune_gpt20b_gguf/causal_lm/pytorch-20B_GGUF-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[arctune_gpt20b_gguf/causal_lm/pytorch-20B_GGUF-single_device-inference]

## Result
FAIL — tt-metal EmbeddingBackwardDeviceOperation shape mismatch: grad_tensor_shape[2] != index_tensor_shape[0] * index_tensor_shape[-1]

## Failure
Original CI failure:
```
Fatal Python error: Segmentation fault
```

Reproduced first as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After applying loader fixes from `remediation/arctune-gpt20b-gguf-gpt-oss-gguf-support`:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
caused by (in C++ worker thread):
```
TT_FATAL @ ttnn/operations/embedding_backward/device/embedding_backward_device_operation.cpp:67:
  grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]
  info: Number of rows in gradient tensor must be equal to number of indices in index tensor
```

## Root cause

Three distinct issues, ordered by fix depth:

1. **Loader — transformers 5.x `model_to_load` kwarg (fixed in remediation branch)**
   Other GGUF loaders (tvall43, gpt_oss_swallow, unified_reward_flex_qwen35, qwen_3_5_imatrix,
   eva_qwen3_next) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
   at module import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers
   5.x now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`. When pytest collects all
   tests it imports all loader modules, so the patched (old) function is live when arctune runs,
   causing TypeError.

2. **Loader — gpt-oss GGUF architecture unregistered (fixed in remediation branch)**
   The GGUF file for arctune-gpt20b declares architecture `gpt-oss`, which transformers 5.x
   does not recognise. Without registration of `gpt-oss` as an alias for `qwen3_moe` (including
   `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, `GGUF_TO_FAST_CONVERTERS`,
   and `TENSOR_PROCESSORS`), the model fails to load or loads with incorrect architecture.

3. **Loader — grouped_mm segfault on CPU (fixed in remediation branch)**
   `torch.nn.functional.grouped_mm` crashes on CPU during
   `partition_fx_graph_for_cpu_fallback`. Forcing `experts_implementation="eager"` selects
   the loop-based MoE forward path instead, avoiding the crash.

4. **Runtime (tt-metal) — EmbeddingBackwardDeviceOperation shape assertion (unfixed)**
   After the three loader fixes are applied, the test reaches actual compilation/execution on
   device and fails with a shape-mismatch assertion inside tt-metal's
   `EmbeddingBackwardDeviceOperation::validate_on_program_cache_miss`:
   ```
   grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]
   ```
   The Qwen3 MoE model (which arctune-gpt20b uses) contains a scatter-over-embedding-table
   operation that StableHLO→TTIR converts to `EmbeddingBackwardOp`. The tt-mlir TTIRToTTNN
   conversion (`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:390`) reshapes the gradient tensor
   to `[1, 1, R, C]` where `R = product of all dims except last`. The assertion then checks
   `R == index[0] * index[-1]`. For the shapes produced by the Qwen3 MoE scatter—which comes
   from tied-weight output-head backpropagation during XLA graph tracing—the dimensions do not
   satisfy this constraint, causing an INTERNAL error code 13 from PJRT/tt-metal.

## Fix

**Loader fixes** (tt-forge-models, branch `remediation/arctune-gpt20b-gguf-gpt-oss-gguf-support`):
- `80ea0240ff`: Changed all `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrappers
  to `(*args, **kwargs)` in eva_qwen3_next, gpt_oss_swallow, tvall43_qwen3_5_2b, qwen_3_5_imatrix
  loaders so the new `model_to_load` kwarg passes through to the original.
- `e32c847118`: Added `_patch_gpt_oss_support()` to the arctune loader registering `gpt-oss` in
  `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, `GGUF_TO_FAST_CONVERTERS`,
  `TENSOR_PROCESSORS`, and `GGUF_CONFIG_DEFAULTS_MAPPING` (where present), pointing to qwen3_moe.
- `d388df4db1`: Set `experts_implementation="eager"` as a default in `load_model()` to avoid the
  CPU grouped_mm crash during partition_fx_graph_for_cpu_fallback.

**Runtime fix** (proposed, in tt-metal):
The fix would live in:
  `ttnn/cpp/ttnn/operations/embedding_backward/device/embedding_backward_device_operation.cpp`

The assertion at line 67 enforces `grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]`.
The shapes that tt-mlir generates for Qwen3 MoE's tied-weight scatter don't satisfy this because
the index tensor's first dimension does not equal 1 (batch folded into first axis vs. last axis).

Proposed fix: In `TTIRToTTNN::EmbeddingBackwardOpConversionPattern::matchAndRewrite`
(`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:390`), after the squeeze step, ensure the index tensor
is reshaped to `[1, total_tokens]` (collapsing all dimensions except the last into a single batch
row), matching the `R` computed for the grad tensor. Alternatively, relax the tt-metal assertion
to check `R == product of all index dims` instead of `index[0] * index[-1]`.

## Verification
Test exited FAILED after 13m01s with `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.
Hardware: n300.
Loader fixes verified to unblock the TypeError and segfault; compiler-stack bug is the remaining blocker.

## Files changed
- `tt-forge-models/arctune_gpt20b_gguf/causal_lm/pytorch/loader.py` (remediation branch commits above)
- `tt-forge-models/eva_qwen3_next_gguf/causal_lm/pytorch/loader.py` (commit 80ea0240ff)
- `tt-forge-models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (commit 80ea0240ff)
- `tt-forge-models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py` (commit 80ea0240ff)
- `tt-forge-models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py` (commit 80ea0240ff)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d388df4db1360a9ff9cf53c33ee1480b4cab1d8b |
