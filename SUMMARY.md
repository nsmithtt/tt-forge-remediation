# Remediation Summary: apexlearn_becoach_qwen3_5_4b_4bit_vlm_mlx-image_text_to_text-pytorch-BECoach_Qwen3_5_4B_4bit_vlm_mlx-single_device-inference

## Skill version
9

## Test
tests/runner/test_models.py::test_all_models_torch[apexlearn_becoach_qwen3_5_4b_4bit_vlm_mlx/image_text_to_text/pytorch-BECoach_Qwen3_5_4B_4bit_vlm_mlx-single_device-inference]

## Result
FAIL — TT device does not support `.tolist()` (eager tensor read) on device tensors; Error code: 13 in Qwen3.5 VisionModel

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Traceback (relevant portion):
```
transformers/models/qwen3_5/modeling_qwen3_5.py:1239: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
transformers/models/qwen3_5/modeling_qwen3_5.py:1162: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Two distinct issues were found:

### Issue 1 (loader — fixed): MLX quantization_config incompatible with transformers
The model `apexlearn/BECoach-Qwen3.5-4B-4bit-vlm-mlx` stores an MLX-format quantization
config in `config.json`. Transformers' `AutoHfQuantizer.supports_quant_method()` requires
a `quant_method` attribute on the quantization config, which the MLX format omits:
```
ValueError: The model's quantization config from the arguments has no `quant_method`
attribute. Make sure that the model has been correctly quantized
```
This is a loader-layer bug analogous to broken `from_single_file` heuristics on quantized
checkpoints. Fixed by stripping `quantization_config` from the loaded config before
`from_pretrained`, causing the model to load in bfloat16 without quantization.

### Issue 2 (runtime — unfixed): `.tolist()` on TT device tensor fails with Error code: 13
After the loader fix, the test runner moves all inputs including `image_grid_thw` (a
`[1, 3]` integer tensor of image grid dimensions) to TT device. `Qwen3_5VisionModel.
fast_pos_embed_interpolate` then calls `grid_thw.tolist()` for Python control flow to
determine position embedding sizes. TT device does not support synchronous eager reads:
attempting `.tolist()` triggers a device synchronization that fails with Error code: 13.

This matches the pattern "int → CPU transfer unsupported" listed as a compiler-stack bug
in the remediation rules.

## Fix

### Applied (loader fix)
- Stripped MLX `quantization_config` from the config before `from_pretrained` in
  `apexlearn_becoach_qwen3_5_4b_4bit_vlm_mlx/image_text_to_text/pytorch/loader.py`.
- Commit: `1d1c0d5f0d` on `remediation/apexlearn-becoach-qwen35-4b-vlm-mlx` in
  `tenstorrent/tt-forge-models`.

### Not applied (compiler-stack bug)
The `.tolist()` failure requires either:
1. **tt-metal/tt-xla**: Support synchronous tensor reads (`.tolist()`) on TT device,
   allowing Python control-flow operations to read back integer scalars from the device.
2. **tt-xla test runner**: Detect tensors used exclusively for Python control flow (e.g.
   integer shape/index tensors not connected to the computation graph) and keep them on
   CPU rather than moving them to TT device with all other inputs.

An existing branch `remediation/apexlearn-becoach-qwen35-4b-vlm-tolist-fix` in
`tenstorrent/tt-forge-models` papers over this with monkey-patching model methods AND
with the forbidden workaround of switching to text-only inputs to avoid a subsequent
VisionEncoder Conv3d L1 OOM. Both are forbidden — they hide the real compiler-stack bugs.

## Verification
pytest exit status: FAILED
Error code: 13 in `Qwen3_5VisionModel.fast_pos_embed_interpolate` at `grid_thw.tolist()`
Hardware: n150 (wormhole_b0)
Wall-clock duration of failing run: 7m 42s

## Files changed
- `apexlearn_becoach_qwen3_5_4b_4bit_vlm_mlx/image_text_to_text/pytorch/loader.py`
  (in `tenstorrent/tt-forge-models`, branch `remediation/apexlearn-becoach-qwen35-4b-vlm-mlx`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 186d3427ac9872321b89f61ccb74e87a5d183eb2 |
| tt-forge-models | 1d1c0d5f0da350ba000b244034de293fce384e60 |
