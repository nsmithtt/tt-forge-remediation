# Remediation Summary: ministral_3b_instruct_bnb_4bit-pytorch-unsloth-Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/ministral_3b_instruct_bnb_4bit/pytorch-unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit-single_device-inference]

## Result
XFAIL — Pixtral vision encoder compilation requires 36 GB DRAM buffer for 2310 patches; exceeds p150b 32 GB capacity

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
pixtral-vision-compilation-dram-overflow-p150b

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
raise ImportError(
    "Using bitsandbytes 4-bit quantization requires bitsandbytes"
)
```

After all 4 loader fixes, the test reaches silicon execution and fails with:
```
Out of Memory: Not enough space to allocate 36440113152 B DRAM buffer across 8 banks
```
(33.9 GB allocation during Pixtral vision tower compilation for a 2310-patch image on p150b with 32 GB DRAM)

The test is correctly marked `KNOWN_FAILURE_XFAIL` in the test config.

## Root cause
Four loader bugs were found and fixed (all in the `loader` layer):

1. **Missing bitsandbytes dependency**: No `requirements.txt` existed; `bitsandbytes` was not installed.

2. **Stale `_dequantize_bnb_4bit` assumed `Params4bit.quant_state`**: bitsandbytes 0.49.2 with `device_map="cpu"` and `torch_dtype=bfloat16` auto-dequantizes weights to plain `Parameter` objects. `_replace_linear4bit` handles both `Params4bit` (with `quant_state`) and plain `Parameter` weights by deriving shape from `weight.shape`.

3. **TT device `prod` reduction bug in `split_sizes` computation**: `Mistral3Model.get_image_features` calls `torch.as_tensor(image_sizes, device=image_features.device).prod(dim=-1)`. On TT device, `prod(dim=-1)` returns wrong results for integer products (e.g., `[[42, 55]].prod(dim=-1)` → 2320 instead of 2310). Patching the function to compute `split_sizes` on CPU (`image_sizes.cpu() // downsample_ratio).prod(dim=-1).tolist()`) avoids the bug.

4. **`aten._local_scalar_dense` in `generate_block_attention_mask`**: The pixtral `generate_block_attention_mask` creates `causal_mask` on the TT device, then uses 0-dim CPU tensors from cumsum as slice bounds (`causal_mask[start:end, start:end]`). Using a 0-dim CPU tensor as a slice bound on a TT device tensor triggers `aten._local_scalar_dense`, which the TT backend cannot compile. Building the mask on CPU and moving to device at the end avoids the issue.

After all 4 fixes the model loads correctly and the test reaches hardware execution, where it hits the genuine hardware capacity ceiling: the Pixtral vision encoder with a 1176×1540 image (2310 patches at 28-pixel downsample ratio) requires a 36 GB DRAM allocation during compilation, exceeding the p150b's 32 GB DRAM.

## Fix
**Loader fixes** in `tt-xla/third_party/tt_forge_models`, branch
`remediation/ministral_3b_instruct_bnb_4bit-pytorch-unsloth-Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit-single_device-inference`:

- `mistral/ministral_3b_instruct_bnb_4bit/pytorch/requirements.txt` (new): Added `bitsandbytes>=0.46.1`.
- `mistral/ministral_3b_instruct_bnb_4bit/pytorch/loader.py`:
  - `_replace_linear4bit`: replaces all `Linear4bit` layers with `nn.Linear`; handles both `Params4bit` (calls `dequantize_4bit`) and auto-dequantized plain `Parameter` (uses weight directly).
  - `_patch_get_image_features`: patches `Mistral3Model.get_image_features` to compute `split_sizes` on CPU; replicates `@merge_with_config_defaults` for `vision_feature_layer=None`; filters duplicate kwargs.
  - `_patch_generate_block_attention_mask`: patches `pixtral.generate_block_attention_mask` to build the causal mask on CPU (avoids `aten._local_scalar_dense` on TT device slice bounds).

**Test config** in `tt-xla` remediation branch: `mistral/ministral_3b_instruct_bnb_4bit/pytorch-unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit-single_device-inference` marked `KNOWN_FAILURE_XFAIL`.

## Verification
- pytest exit: xfailed (KNOWN_FAILURE_XFAIL, as expected)
- Hardware: blackhole-p150b
- Duration: 208.44s (0:03:28)
- Tier A attempts: N/A

## Files changed
- `mistral/ministral_3b_instruct_bnb_4bit/pytorch/requirements.txt` (created)
- `mistral/ministral_3b_instruct_bnb_4bit/pytorch/loader.py` (modified)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | be577b5b940d137752e83ae79167b08d99b15fb3 |
| tt-forge-models | 1373332709d7d0019242d6223c28c3bb0efc8076 |
