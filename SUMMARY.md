# Remediation Summary: intern_vl_3_5_gguf-image_to_text-pytorch-14b_q4_k_m-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[intern_vl_3_5_gguf/image_to_text/pytorch-14b_q4_k_m-single_device-inference]

## Result
FAIL — Tier B compiler bug: ttnn::cumsum receives a 37.9 GB input tensor from masked_scatter lowering (ttmlir-cumsum-shape-overflow-masked-scatter)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-cumsum-shape-overflow-masked-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Immediate cause (from device log):
```
TT_FATAL: Out of Memory: Not enough space to allocate 37958451200 B DRAM buffer
across 8 banks, where each bank needs to store 4744806400 B, but bank size is
4273390016 B (allocated: 3838589504 B, free: 434800512 B, largest free block:
395120320 B)
```

Stack at OOM:
```
ttnn::cumsum → accumulation_invoke → preprocess_input_tensor → ttnn::permute
→ PermuteDeviceOperation::create_output_tensors → allocate_device_buffer → FATAL
```

## Root cause

Three bugs were fixed in the loader before silicon execution was reached:

**Bug 1 (loader):** The original loader called `AutoModelForImageTextToText.from_pretrained(..., gguf_file=...)`. The bartowski GGUF has `general.architecture: 'qwen3'`, so transformers creates `Qwen3Config` and rejects it via `AutoModelForImageTextToText`. Transformers also has no native `internvl` GGUF support (`internvl` is not in `GGUF_SUPPORTED_ARCHITECTURES`). The loader was rewritten to:
1. Load the Qwen3 LLM from the main GGUF via `AutoModelForCausalLM`
2. Remap its state dict keys (`model.*` → `model.language_model.*`)
3. Parse the mmproj GGUF (vision encoder, clip-arch, F16) with `GGUFReader`
4. Build a bare `InternVLForConditionalGeneration` with `init_empty_weights()` and load the combined state dict via `load_state_dict(strict=True, assign=True)`

**Bug 2 (loader, cross-session contamination):** 26 qwen35/gpt-oss GGUF loaders define `_patched_load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)`. These patches are installed at module import time; the last-imported one replaces the real `transformers.load_gguf_checkpoint`. When transformers 5.2.0 added `model_to_load=dummy_model` to its call, any test loaded after one of those 26 modules raised `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Fixed by adding `*args, **kwargs` to all 26 patch definitions and their inner forwarding calls.

**Terminal bug (tt-mlir, Tier B):** After all loader fixes, the model compiles successfully but fails during execution. `InternVLForConditionalGeneration.forward` at `modeling_internvl.py:655` runs:

```python
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

where `inputs_embeds` is `[1, 1810, 5120]` and `special_image_mask` is the boolean expanded mask. The StableHLO→TTIR lowering generates a `CumSumOp` whose input tensor is inflated to `37,958,451,200 B` (~37.9 GB) — far larger than the device DRAM (31.84 GB). `ttnn::cumsum`'s `preprocess_input_tensor` transposes this tensor via `ttnn::permute`, which fails trying to allocate a 37.9 GB output buffer.

This is the same compiler bug as KORMo-VL (fingerprint `ttmlir-cumsum-shape-overflow-masked-scatter`). In both cases, a `masked_scatter(bool_mask, features)` pattern produces an inflated CumSumOp input. For KORMo-VL the tensor was ~46 GiB from a `[1, 2948, 4096]` input; here it is ~35.4 GiB from a `[1, 1810, 5120]` input. The inflation factor matches `seq_rounded × hidden × 2048 × 2` (1824 × 5120 × 2048 × 2 ≈ 38.2 GB), consistent with a tile-size or double-buffer arithmetic bug in the scatter→cumsum lowering path in tt-mlir.

## Fix
**Loader fixes (committed on `remediation/intern_vl_3_5_gguf-image_to_text-pytorch-14b_q4_k_m-single_device-inference` in tt_forge_models):**
- `intern_vl_3_5_gguf/image_to_text/pytorch/loader.py` — complete rewrite of `load_model` implementing split-GGUF loading; added `_MMPROJ_FILES`, `_HF_PROCESSORS` class variables
- 26 × `*/causal_lm/pytorch/loader.py` — narrowed `_patched_load_gguf_checkpoint` signature widened to accept `*args, **kwargs`

**Proposed compiler fix (not attempted):** The `masked_scatter` → cumsum inflation likely lives in `StableHLOLegalizeCompositePass.cpp` or the TTIR→TTNN lowering for `scatter_nd`/`select_and_scatter`. The fix would be to correctly compute the output shape for the intermediate CumSumOp from the scatter input shape, avoiding the tile-size multiplication. Root cause diagnosis requires dumping the TTIR flatbuffer and tracing which lowering step produces the inflated shape.

## Tier B justification
**Indicator:** `internal-error-unknown-mechanism` / `cross-cutting`

The CumSumOp shape inflation has the same fingerprint as KORMo-VL (filed Tier B) with no identified fix location. Diagnosing it requires dumping and inspecting the TTIR flatbuffer to trace which lowering step inflates the shape — that diagnosis-first work is outside the single-PR scope of a Tier A fix. The bug affects any model using `masked_scatter` with a boolean mask in its forward pass, making it cross-cutting.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 583.55s (0:09:43)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/intern_vl_3_5_gguf/image_to_text/pytorch/loader.py` (complete rewrite)
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 6d5d796087922e0e2018cde0c1d32ae6a0d08c26 |
