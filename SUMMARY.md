# Remediation Summary: huihui_qwen_3_5_0_8b_abliterated_i1_gguf-causal_lm-pytorch-0_8B_Abliterated_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen_3_5_0_8b_abliterated_i1_gguf/causal_lm/pytorch-0_8B_Abliterated_i1_GGUF-single_device-inference]

## Result
FAIL — loader fixed (qwen35 GGUF arch registration); residual PCC=0.797 on WH silicon is below the WH BF16 matmul precision floor threshold; Tier B compiler-stack precision bug.

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(Cross-loader contamination from narrow-sig patchers installed by other GGUF loaders at collection time, combined with missing qwen35 arch registration in this loader.)

After loader fix:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7972252978615758. Required: pcc=0.99.
```

## Root cause
Two bugs:

**Bug 1 (loader, fixed):** The loader was missing the qwen35 GGUF architecture registration entirely. The GGUF file declares `general.architecture = qwen35` and uses a hybrid SSM+attention architecture (`qwen3_5_text` in transformers) with `full_attention_interval=4`. Four tables must be populated for transformers to load qwen35 GGUFs: `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"]`, `TENSOR_PROCESSORS["qwen35"]`, and `GGUF_TO_FAST_CONVERTERS`. Without these, the load raises `ValueError: GGUF model with architecture qwen35 is not supported yet`.

Additionally, transformers 5.2.0 added a `model_to_load=` kwarg to `load_gguf_checkpoint`. Twenty-six other GGUF loaders in the test suite install narrow-signature patchers at import time. These crash with `TypeError` when transformers calls them with `model_to_load=`. The fix uses a DFS chain-walk (`_find_real_load_gguf_checkpoint`) to locate the real function through any installed patchers, then installs a qwen35-aware patcher using `*args, **kwargs` that survives the transformers 5.2.0 call signature.

Critical SSM metadata mappings required: `ssm.time_step_rank -> linear_num_value_heads` (value=16), `ssm.state_size -> linear_key_head_dim`, `ssm.group_count -> linear_num_key_heads`, `ssm.conv_kernel -> linear_conv_kernel_dim`. The `Qwen35TensorProcessor` must expand `ssm_conv1d.weight` from 2D to 3D. The patcher must remap `model_type: qwen35 -> qwen3_5_text` and derive `layer_types` from `full_attention_interval`.

**Bug 2 (tt-mlir, not fixed):** After the loader fix the model runs on WH silicon but produces PCC=0.797, far below the required 0.99. The 2B variant of the same qwen3_5_text architecture gives PCC=0.884 on WH (also below threshold). The 0.8B model gives an even lower PCC, consistent with smaller models accumulating larger relative BF16 rounding errors. The qwen3_5_text architecture has 24 layers: 6 full-attention and 18 GatedDeltaNet (linear attention) SSM layers. The BF16 precision degradation in tt-mlir matmuls compounds across both layer types. This is the same `ttmlir-bf16-matmul-precision-floor` bug documented for the 2B variant and multiple other Wormhole models. No F32 precision path is available on WH silicon for models of this size.

## Fix
**Loader fix** in `tt-xla/third_party/tt_forge_models/huihui_qwen_3_5_0_8b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`:

- Added `_register_qwen35_gguf_tables()` called at import time: populates all four transformers GGUF tables for qwen35 architecture with correct SSM key mappings.
- Added `Qwen35TensorProcessor`: expands `ssm_conv1d.weight` from 2D `[D,K]` to 3D `[D,1,K]`.
- Added `_find_real_load_gguf_checkpoint()`: DFS chain-walk to find the real function past any narrow-sig patchers.
- Added `_build_qwen35_patcher()`: wraps real function with `*args, **kwargs`; remaps `model_type: qwen35 -> qwen3_5_text` and derives `layer_types` list from `full_attention_interval`.
- Added `_qwen35_load_ctx()`: context manager that installs the patcher for `_gguf_utils`, `_config_utils`, `_auto_tokenizer` plus patches `get_gguf_hf_weights_map` to remap `qwen3_5_text -> qwen35` for weight mapping.
- Added `model_kwargs.setdefault("use_cache", False)` to prevent `Qwen3_5DynamicCache` TypeError.

**Proposed fix for Bug 2:** The `ttmlir-bf16-matmul-precision-floor` bug requires adding an F32 accumulation path for matmuls in `tt-mlir` on Wormhole silicon. This is a cross-cutting change and is Tier B.

## Tier B justification
cross-cutting — fixing BF16 matmul precision on WH requires enabling F32 accumulation (or higher math fidelity) across all matmul lowerings in tt-mlir, touching every matmul kernel path. This is not a scoped single-function fix.

## Verification
- pytest exit: FAIL
- Hardware: wormhole
- Duration: 2541.32s (0:42:21)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huihui_qwen_3_5_0_8b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` — complete rewrite with qwen35 GGUF arch registration and load context manager

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | cb738cebcccba4e3b234bc57e4bb158315f36d97 |
| tt-forge-models | 6e9e8b856c28150deb1507faa831379c56d3687f |
