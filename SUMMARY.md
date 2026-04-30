# Remediation Summary: glm_4_7_flash_claude_opus_distill_gguf-causal_lm-pytorch-4_7_Flash_Claude_Opus_4_5_High_Reasoning_Distill_v2_heretic_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_claude_opus_distill_gguf/causal_lm/pytorch-4_7_Flash_Claude_Opus_4_5_High_Reasoning_Distill_v2_heretic_i1_GGUF-single_device-inference]

## Result
XFAIL — GLM-4.7-Flash architecture: 64 routed experts, 47 layers, ~32B params (~64 GB BF16 after GGUF dequantization) exceeds single-device DRAM on all supported hardware (12 GB n150, 24 GB p150b)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-7-flash-claude-opus-distill-gguf-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
OSError: Can't load the configuration of 'mradermacher/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-v2-heretic-i1-GGUF'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'mradermacher/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-v2-heretic-i1-GGUF' is the correct path to a directory containing a GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-v2-heretic.i1-Q4_K_M.gguf file
```
Preceded by:
```
UserWarning: Not enough free disk space to download the file. The expected file size is: 18132.72 MB.
```
```
RuntimeError: Data processing error: File reconstruction error: Internal Writer Error: Background writer channel closed
```

## Root cause
**Layer: hardware-class**

The GGUF file `GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-v2-heretic.i1-Q4_K_M.gguf`
is 18,132 MB at Q4_K_M quantization. The base model is GLM-4.7-Flash with architecture
`glm4_moe_lite`: 47 layers, 64 routed experts + 1 shared expert, hidden_size=2048,
num_experts_per_tok=4. At Q4_K_M (~4.5 bits/param), 18 GB corresponds to ~32B parameters.
After dequantization to BF16 (required for TT device inference), the model occupies ~64 GB —
exceeding the DRAM of all supported single-device configurations (n150: 12 GB, p150b: 24 GB).

The immediate test failure was the loader's `load_config()` method calling
`AutoConfig.from_pretrained('mradermacher/...', gguf_file='...')` which triggered a download
of the 18 GB GGUF file. The disk had insufficient free space (< 8 GB), causing the download to
fail with a "Background writer channel closed" error, then an OSError when config loading aborted.
Even if disk space were available, the model cannot fit on a single TT device at BF16 precision.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla, near the
other `glm` entries (around line 2636), with reason string documenting the hardware capacity ceiling.

## Verification
- pytest exit: not-run (hardware-class, model cannot be loaded on single device)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a2a6b5a872400c98e2031245a95eafd82fccb476 |
| tt-forge-models | 0f7b734348 |
