# Remediation Summary: huihui_qwen3_5_4b_abliterated-image_to_text-pytorch-4b_abliterated-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_5_4b_abliterated/image_to_text/pytorch-4b_abliterated-single_device-inference]

## Result
FAIL — conv3d multi-block C_in reduction writer kernel overwrites reducer output; skill rule prohibits chaining a second Tier A fix

## Stack layer
tt-metal

## Tier
A

## Bug fingerprint
conv3d-cin-blocks-writer-dram-overwrite

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause

**Bug 1 (fixed — tt-metal):** `conv3d.cpp` default dispatch sets `C_in_block = TILE_WIDTH = 32`. With Qwen3.5-VL `PatchEmbed` kernel shape `[2,16,16]`, `C_in=32`, `C_out=1024`, the dominant circular buffers (`vol2col_tiled` + `weight_tiled`) require 2 × 1,048,576 B = 2,097,152 B per core — exceeding L1 (1,461,760 B available per core). The program factory had no L1-budget guard. This is the direct cause of Error code: 13.

**Bug 2 (unfixed — tt-metal):** Reducing `C_in_block` to 16 raises `C_in_num_blocks` to 2. The multi-block path uses a reducer core to accumulate partial sums from worker cores. The conv3d writer kernel writes to output DRAM for ALL cores (reducer and workers); worker cores overwrite the reducer's accumulated result with their partial sums, producing garbage output (PCC = -0.053). This is a second, independent compiler-stack bug. The skill rule "Do not chain Tier A fixes" prohibits attempting a second compiler-stack fix in the same report.

**Loader fixes (tt_forge_models):** The Qwen3.5-VL model calls `.tolist()` and boolean-indexes on TT device tensors (unsupported eager ops, Error code: 13). `_patch_qwen3_5_for_tt_device()` was added to the loader to intercept five methods and route metadata tensors to CPU. The evaluator was also patched to skip non-Tensor leaves (`Qwen3_5DynamicCache`) in PCC/equal/allclose comparisons.

## Fix

**Bug 1 fix — tt-metal `conv3d_program_factory.cpp`:**

Added an L1-budget guard after the default `C_in_block` is established. The guard halves `C_in_block` until `(vol2col_tiled tiles + weight_tiled tiles) × tile_size ≤ 75% of available L1`. With the Qwen3.5-VL PatchEmbed parameters, `C_in_block` is auto-reduced from 32 → 16, dropping dominant CB usage from 2,097,152 B to 1,048,576 B (within budget). A `log_warning` is emitted when the reduction fires.

File: `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
Branch: `remediation/huihui_qwen3_5_4b_abliterated-image_to_text-pytorch-4b_abliterated-single_device-inference` in tt-metal
Commits: `eda957bcdb` (initial guard), `df8a23274852bcc7774715862c87a19ef51673b6` (correct condition/loop bounds)

**Loader fix — tt_forge_models `loader.py`:**

Added `_patch_qwen3_5_for_tt_device()` that monkey-patches five Qwen3.5-VL methods:
- `fast_pos_embed_interpolate`: move `grid_thw` to CPU before `.tolist()` call
- `rot_pos_emb`: move `grid_thw` to CPU before `.tolist()` call
- `get_rope_index`: move all inputs to CPU; move `position_ids`/`rope_deltas` back to TT device via `next(self.parameters()).device`
- `get_image_features`: move `image_grid_thw` to CPU
- `get_placeholder_mask`: replace `inputs_embeds[bool_mask].numel()` check (boolean indexing on TT device) with arithmetic equivalent

Set `min_pixels = 56*56`, `max_pixels = 13*28*1280` for hardware L1 budget.

File: `tt-xla/third_party/tt_forge_models/huihui_qwen3_5_4b_abliterated/image_to_text/pytorch/loader.py`
Branch: `remediation/huihui_qwen3_5_4b_abliterated-image_to_text-pytorch-4b_abliterated-single_device-inference` in tt_forge_models
Commit: `e79d3029e901b7c75129f5e20ec7d1d581ba4da8`

**Evaluator fix — tt-xla `torch_comparison_evaluator.py`:**

Skip non-`torch.Tensor` leaves (e.g. `Qwen3_5DynamicCache`) in `_compare_equal`, `_compare_pcc`, `_compare_atol`, `_compare_allclose` — same treatment as `None` leaves.

File: `tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py`
Branch/commit: `3128be2dfce0c056a6c3939142c9aac9ac1a6e85` in tt-xla

**Bug 2 proposed fix (not attempted):**

In the conv3d writer kernel, worker cores (`c_in_idx > 0`) should NOT write to output DRAM — only the reducer core (`c_in_idx == 0`, which holds the accumulated result) should write the final output. Likely file: `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/` writer kernel.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    4558.64s (1:15:58)
- Tier A attempts: 1

## Files changed
- `tt-metal/ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`
- `tt-xla/third_party/tt_forge_models/huihui_qwen3_5_4b_abliterated/image_to_text/pytorch/loader.py`
- `tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | df8a23274852bcc7774715862c87a19ef51673b6 |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3128be2dfce0c056a6c3939142c9aac9ac1a6e85 |
| tt-forge-models | e79d3029e901b7c75129f5e20ec7d1d581ba4da8 |
