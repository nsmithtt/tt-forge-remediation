# Remediation Summary: mistral_small_3_1_24b_instruct_2503_quantized_w4a16

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_small_3_1_24b_instruct_2503_quantized_w4a16/pytorch-24B_Instruct_2503_Quantized_W4A16-single_device-inference]

## Result
XFAIL — 24B model dequantized from INT4 to BF16 (~48 GB) exceeds p150b single-device DRAM (~34 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-24b-bf16-dram-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
The image processor of type `PixtralImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

Actual first failure on this branch (before loader fixes):
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors
quantization. Please install it with `pip install compressed-tensors`.
```

Terminal failure after all loader fixes:
```
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each
bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4215689216 B, free:
57700800 B, largest free block: 39949952 B)
```

## Root cause
The model is `RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16`, a 24B parameter
VLM with W4A16 (4-bit weights, 16-bit activations) compression via the `compressed-tensors`
format (pack-quantized INT4). For TT device inference, these INT4 layers must be dequantized
to BF16 as the TT backend has no compressed-tensors kernel support.

After dequantization, the model is approximately 24B × 2 bytes = 48 GB in BF16. The p150b
device has 8 GDDR banks × 4.27 GB/bank ≈ 34 GB total device DRAM. The model exceeds single-
device capacity by ~14 GB.

The loader required five additional fixes before reaching the hardware ceiling:
1. `compressed-tensors` package was missing from `requirements.txt`.
2. The quantization config `ignore` list stored pre-transformers-5.x flat paths
   (`vision_tower.*`, `multi_modal_projector.*`, `language_model.lm_head`). In transformers 5.x
   these are wrapped under `model.*`, causing `compressed-tensors` to wrongly quantize the vision
   tower and projector (creating `weight_packed` attributes the checkpoint doesn't have).
3. `AutoProcessor.from_pretrained` in transformers 5.x loads `PixtralImageProcessor` as the fast
   variant by default; `use_fast=False` is needed to match the saved checkpoint.
4. `get_image_features` computes `split_sizes` on the TT device where int64 arithmetic promotes to
   bfloat16; large patch counts (e.g. 2310) round to incorrect values. Fix: force CPU computation.
5. `generate_block_attention_mask` uses in-place XLA tensor assignment inside a Python loop
   which causes a Dynamo graph break and INTERNAL Error code 13 on TT.

## Fix
**Loader fixes in `tt_forge_models` (remediation branch)**:
- `mistral_small_3_1_24b_instruct_2503_quantized_w4a16/pytorch/requirements.txt`: add `compressed-tensors`
- `mistral_small_3_1_24b_instruct_2503_quantized_w4a16/pytorch/loader.py`:
  - Fix quantization ignore list paths to match transformers 5.x `model.*` structure
  - Dequantize pack-quantized INT4 layers via `PackedQuantizationCompressor.decompress()` after `from_pretrained`
  - Add `use_fast=False` to `AutoProcessor.from_pretrained()`
  - Patch `Mistral3Model.get_image_features` to compute `split_sizes` on CPU
  - Patch `generate_block_attention_mask` with functional version (no in-place XLA mutation)
  - Fix `load_shard_spec` paths: `model.language_model.layers` (not `.model.layers`) and Pixtral attention/FFN attribute names

**Test config** (`tt-xla`): Add `KNOWN_FAILURE_XFAIL` entry for this test with hardware-class OOM reason.

## Verification
- pytest exit: FAIL (OOM after all loader fixes — hardware-class ceiling, not compiler bug)
- Hardware:    blackhole-p150b
- Duration:    545.18s (0:09:05) including model load + dequantization
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)
- `tt-xla/third_party/tt_forge_models/mistral_small_3_1_24b_instruct_2503_quantized_w4a16/pytorch/loader.py` (five loader fixes)
- `tt-xla/third_party/tt_forge_models/mistral_small_3_1_24b_instruct_2503_quantized_w4a16/pytorch/requirements.txt` (new: compressed-tensors)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 05b1f437b40ccac2fe859b8e9bbbec3c16280e20 |
| tt-forge-models | 15526bfc8700f6d10e2e4ea144a1c9542ac55768 |
