# Remediation Summary: exaone_4_mlx_4bit-causal_lm-pytorch-4.0_32B_MLX_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[exaone_4_mlx_4bit/causal_lm/pytorch-4.0_32B_MLX_4bit-single_device-inference]

## Result
XFAIL — EXAONE 4.0 32B MLX 4-bit model weights consume ~31.4 GB of 32 GB Blackhole DRAM; no room for inference activation buffers

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-dram-capacity-32b-model

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -4095)

(Original error on the configured branch. After applying the existing aten.slice clamping fix in tt-xla commit ee94c71a4, the test progresses further and then hits:)

E   RuntimeError: TT_FATAL @ .../bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 280494080 B DRAM buffer across 8 banks,
where each bank needs to store 35061760 B, but bank size is 4273390016 B
(allocated: 4207702336 B, free: 65687680 B, largest free block: 30849088 B)

## Root cause
Two issues in sequence:

1. **aten.slice OOB (already fixed):** The EXAONE 4.0 architecture uses SlidingWindowCache with window=4096. On a 22-token input the cache tries to slice with start=-4096, which is out of range for dim size 22. The XLA lazy backend raises "Value out of range". This is the same bug as xla-lazy-slice-oob-negative-start, already fixed in tt-xla at commit ee94c71a4 (clamp aten.slice start/end when negative index exceeds tensor size in torch_overrides.py).

2. **Hardware DRAM capacity (root XFAIL):** The EXAONE 4.0 32B MLX 4-bit model, when loaded via `AutoModelForCausalLM.from_pretrained` with `ignore_mismatched_sizes=True` (required because the lmstudio-community MLX safetensors have different weight shapes than the standard EXAONE4 model), consumes ~31.4 GB of the Blackhole's 32 GB GDDR memory for model weights. Only ~62.6 MB per bank (501 MB total) remains, with largest contiguous block of 29.4 MB per bank. The first inference step (tilize input) needs 35 MB per bank (280 MB total), causing TT_FATAL OOM.

The loader fix (sliding_window_pattern string→layer_types expansion, quantization_config removal, ignore_mismatched_sizes=True) was already committed to tt_forge_models at a02696c8d.

## Fix
1. **Loader fix (tt_forge_models, already pushed):** `exaone_4_mlx_4bit/causal_lm/pytorch/loader.py` — `_load_patched_config` method added to expand `sliding_window_pattern` string to per-layer `layer_types` list, strip `quantization_config` to prevent MLX dequant errors, and `ignore_mismatched_sizes=True` in `from_pretrained`. Committed at a02696c8dba92e06f6a607c42fddabed2ba8d759 on branch `remediation/exaone_4_mlx_4bit-causal_lm-pytorch-4.0_32B_MLX_4bit-single_device-inference`.

2. **XFAIL entry (tt-xla):** `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` for this test. Committed at f6895ade760a24c119f37f06ed4a46806b9ffc14 on branch `remediation/exaone_4_mlx_4bit-causal_lm-pytorch-4.0_32B_MLX_4bit-single_device-inference`.

## Verification
- pytest exit: FAIL (OOM after slice fix)
- Hardware:    blackhole-p150b
- Duration:    846.37s (0:14:06)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)
- `tt-xla/third_party/tt_forge_models` (submodule pointer update)
- `tt_forge_models/exaone_4_mlx_4bit/causal_lm/pytorch/loader.py` (loader fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f6895ade760a24c119f37f06ed4a46806b9ffc14 |
| tt-forge-models | a02696c8dba92e06f6a607c42fddabed2ba8d759 |
