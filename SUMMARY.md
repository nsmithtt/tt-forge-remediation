# Remediation Summary: gemma3_12b_it_gptq_4b_128g-multimodal-pytorch-ISTA-DASLab-gemma-3-12b-it-GPTQ-4b-128g-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_12b_it_gptq_4b_128g/multimodal/pytorch-ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g-single_device-inference]

## Result
FAIL — 12B BF16 dequantized model (≈24 GB weights) + activation buffers exceed p150b DRAM; int4 quantized execution is not supported in tt-mlir

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-int4-quantized-execution-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Device log: `Out of Memory: Not enough space to allocate 4356833280 B DRAM buffer across 8 banks, where each bank needs to store 544604160 B, but bank size is 4273390016 B (allocated: 3648197952 B, free: 625192064 B)`

## Root cause
Three loader bugs were fixed before reaching the OOM (see Fix section). The remaining blocker is in tt-mlir: TT silicon has no support for executing GPTQ int4-packed weight formats directly. The loader is forced to call `run_compressed=False` on the `compressed_tensors` quantization config, which dequantizes all weights from int4 (≈6 GB) to BF16 (≈24 GB) at load time. The 12B language model weights (24 GB BF16) plus the SigLIP vision tower (~900 MB) plus the activation buffers allocated during the `cumsum` op in the forward pass exceed the p150b Blackhole device's available DRAM (approximately 34 GB total, with ≈29.5 GB consumed before the OOM). The model would fit in int4 (6 GB), but there is no int4 matmul kernel path in tt-mlir to execute it.

## Fix
Three loader-layer fixes were applied in `tt_forge_models` on branch `remediation/gemma3_12b_it_gptq_4b_128g-multimodal-pytorch-ISTA-DASLab-gemma-3-12b-it-GPTQ-4b-128g-single_device-inference`:

1. **`use_fast=False`** in `AutoProcessor.from_pretrained()` — transformers 5.x changed the default image processor for Gemma3 from slow to fast; passing `use_fast=False` restores the original behavior.

2. **`compressed_tensors` ignore-pattern fix** — compressed_tensors 0.15.x uses `re.match` (anchored at string start) for ignore patterns. The checkpoint's ignore list contains `"re:vision_tower.*"` which fails to match `"model.vision_tower.*"` module paths (since `Gemma3ForConditionalGeneration` wraps everything under `self.model`). Fix: rewrite all `re:` prefixed patterns to `re:.*` (prepend `.*`). Also added `"lm_head"` to the ignore list to cover the top-level `lm_head` (tied to `embed_tokens.weight`) which is distinct from `"language_model.lm_head"` in the checkpoint's existing ignore list, and set `run_compressed=False` to dequantize weights to float.

3. **`load_shard_spec` attribute paths** — the original code accessed `model.vision_tower` and `model.language_model.layers`, but `Gemma3ForConditionalGeneration` has `self.model = Gemma3Model` which holds `vision_tower` and `language_model`. Corrected to `model.model.vision_tower.vision_model.encoder.layers` and `model.model.language_model.layers`.

One Tier A compiler-stack fix was applied in `tt_xla` on branch `remediation/gemma3_12b_it_gptq_4b_128g-multimodal-pytorch-ISTA-DASLab-gemma-3-12b-it-GPTQ-4b-128g-single_device-inference`:

4. **`normalize_slice_indices` graph pass** — `transformers.SlidingWindowCache.update` emits `full_value_states[:, :, -self.sliding_window + 1:, :]` (i.e., start index -1023 for `sliding_window=1024`). When the cache tensor's dim 2 has only 277 elements, this index is out of the range `[-277, 276]` that XLA accepts. PyTorch CPU silently clamps it but XLA raises "Value out of range". The pass runs after `run_decompositions` where `node.meta['val']` has concrete shapes, and rewrites any `aten.slice.Tensor` start index satisfying `start < -dim_size` to `max(0, dim_size + start)`.

The OOM error remained after all four fixes. No further fix is attempted because the root cause requires Tier B infrastructure work in tt-mlir.

## Tier B justification
**Indicator: new-infrastructure**
Executing GPTQ int4-packed weight tensors requires implementing int4 matmul kernel paths in tt-mlir — there is no existing infrastructure for dequantize-on-the-fly or int4 GEMM in the TT compiler stack. Without it, the loader must inflate weights to BF16 at load time, causing the OOM.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    445.43s
- Tier A attempts: 1

## Files changed
**tt_forge_models** (`remediation/gemma3_12b_it_gptq_4b_128g-multimodal-pytorch-ISTA-DASLab-gemma-3-12b-it-GPTQ-4b-128g-single_device-inference`):
- `gemma3_12b_it_gptq_4b_128g/multimodal/pytorch/loader.py` — use_fast=False, compressed_tensors ignore-pattern fix, lm_head ignore, run_compressed=False, quantized_forward cleanup, corrected load_shard_spec paths
- `gemma3_12b_it_gptq_4b_128g/multimodal/pytorch/requirements.txt` — added `compressed-tensors`

**tt_xla** (`remediation/gemma3_12b_it_gptq_4b_128g-multimodal-pytorch-ISTA-DASLab-gemma-3-12b-it-GPTQ-4b-128g-single_device-inference`):
- `python_package/tt_torch/backend/passes.py` — added `normalize_slice_indices` graph pass
- `python_package/tt_torch/backend/backend.py` — import and call `normalize_slice_indices` in `torch_pass_pipeline`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a85c8926c1a2f0b7223673bef77394ffb5c54941 |
| tt-forge-models | b83c13384932f426d061ac5b4012c4d56a12b4c3 |
