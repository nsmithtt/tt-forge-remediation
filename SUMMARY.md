# Remediation Summary: gpt_oss_gguf-causal_lm-pytorch-20B_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_gguf/causal_lm/pytorch-20B_GGUF-single_device-inference]

## Result
FAIL — second INTERNAL error (error code 13) during _xla_sync_multi execution after loader and first compiler fixes applied

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
qwen3moe-embedding-backward-internal-error-xla-sync

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Fatal Python error: Segmentation fault
```

Later (after loader fixes), the segfault was resolved but a second error appeared:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
  at dynamo_bridge.py:583 in _xla_sync_multi during optimized_mod execution
```

## Root cause

**Error 1 (segfault):** The `grouped_mm` CUDA kernel used by transformers 5.x Qwen3MoE
experts implementation crashes on CPU during `partition_fx_graph_for_cpu_fallback`
when `is_grouped_mm_available()` returns True. Fixed by forcing `experts_implementation="eager"`.

**Error 2 (loader AttributeError):** `load_shard_spec` assumed a dense MLP structure
(`mlp.up_proj`, etc.) but GPT-OSS 20B uses Qwen3MoE with batched expert weights
(`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`). Fixed with `hasattr` guards.

**Error 3 (first INTERNAL/13):** `EmbeddingBackwardDeviceOperation::validate_on_program_cache_miss`
in tt-metal asserts `grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]`.
The `EmbeddingBackwardOpConversionPattern` in `TTIRToTTNN.cpp` was squeezing trailing
dim-1 dimensions from the index tensor producing a 1D shape `[N]`, so
`index[0] * index[-1] = N² ≠ N`. Fixed by adding a reshape to `[1, total_tokens]`
after the squeeze.

**Error 4 (second INTERNAL/13, unfixed):** After the tt-mlir fix, a new INTERNAL error 13
appears at a different call site — `torch_xla._XLAC._xla_sync_multi` during model
execution in `optimized_mod`. The exact failing op is unknown; the INTERNAL code with
no further diagnostic text prevents further analysis without additional tracing
infrastructure. This is Tier B.

## Fix

**Loader fixes (tt_forge_models, `remediation/gpt_oss_gguf-causal_lm-pytorch-20B_GGUF-single_device-inference`):**

1. `gpt_oss_gguf/causal_lm/pytorch/loader.py`:
   - Added `model_kwargs.setdefault("experts_implementation", "eager")` to force the
     CPU-safe MoE implementation and avoid `grouped_mm` segfault.
   - Rewrote `load_shard_spec` with `hasattr` guards for MoE vs dense MLP structure
     to handle Qwen3MoE's `Qwen3MoeExperts` batched weight layout.

2. Cherry-picked commit `7f2c9e8440` (originally from `remediation/davidau-openai-gpt-oss-20b-coder-neo-code-di-matrix-gguf`):
   Changed all 26 affected `_patched_load_gguf_checkpoint` functions from
   `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` to accept the
   `model_to_load` kwarg added in transformers 5.2.0.

**Compiler fix (tt-mlir, `remediation/gpt_oss_gguf-causal_lm-pytorch-20B_GGUF-single_device-inference`):**

`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `EmbeddingBackwardOpConversionPattern::matchAndRewrite`:
After the existing trailing-dim-1 squeeze block, added a reshape of the index tensor
to `[1, total_tokens]` so the tt-metal assertion
`grad_shape[2] == index[0] * index[-1]` holds for any input index rank/shape
(e.g. Qwen3MoE scatter indices arrive as `[batch, seq_len]` instead of `[tokens]`).

**Proposed fix for the remaining FAIL:**

The second INTERNAL error originates in `torch_xla._XLAC._xla_sync_multi` during
compiled graph execution. Root-cause requires enabling verbose tt-xla/tt-mlir
diagnostic logging and a targeted run to isolate the failing StableHLO op. The
fix likely lives in tt-mlir (a missing or incorrect lowering for one of the Qwen3MoE
ops used during the forward pass) but the error code alone is insufficient to identify
the file or function.

## Tier B justification

`internal-error-unknown-mechanism`: The second INTERNAL error (code 13) at
`_xla_sync_multi` provides no op name, tensor shape, or stack trace from the tt-mlir
or tt-metal side. Diagnosis requires enabling verbose runtime tracing (new
infrastructure not present in the current test harness) before the fix can be
identified. The root cause is unknown and cannot be addressed with a scoped
single-file change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1550.15s (0:25:50)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/gpt_oss_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_gguf_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (cherry-pick)
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py` (cherry-pick)
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py` (cherry-pick)
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py` (cherry-pick)
- `tt_forge_models/` (22 additional loaders updated via cherry-pick for model_to_load kwarg)
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 2a3aae5d0454e9064f990a03774af4f598ce1a2d |
| tt-xla          | b8a495547c0a0a36203b5c4a6c44b54b721c3a74 |
| tt-forge-models | 649a5aaa24e9ebe4ac472413be20783f31d20322 |
