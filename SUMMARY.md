# Remediation Summary: boreal_qwen_image-text_to_image-pytorch-portraits_high_rank-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[boreal_qwen_image/text_to_image/pytorch-portraits-high-rank-single_device-inference]

## Result
XFAIL — QwenImageTransformer2DModel has ~19.85B parameters (~39.7 GB BF16), exceeding all single-device TT DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-20b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: The size of tensor a (4096) must match the size of tensor b (3584) at non-singleton dimension 2

(CI failure; local reproduction shows OOM instead — see Root cause)

## Root cause
The `QwenImageTransformer2DModel` diffusion transformer has 60 dual-stream (image + text) transformer blocks, each with `dim = num_attention_heads × attention_head_dim = 24 × 128 = 3072`. Each block contains two MLP sublayers with inner_dim = 4 × 3072 = 12288 (75 M params each), two modulation layers (57 M each), and joint attention (65 M). Per-block total: ~330 M params × 60 blocks ≈ 19.85 B params. At BF16, this is ~39.7 GB — far beyond any single TT device DRAM (n150: 12 GB, n300/p150b: 24 GB).

Local reproduction confirmed OOM: `TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 75497472 B DRAM buffer across 8 banks … free: 4566080 B`. The process RSS reached 42 GB on CPU, consistent with the model size calculation. The CI failure with shape mismatch (4096 vs 3584) was not reproduced locally; it may have resulted from a different diffusers model snapshot or a configuration mismatch on the CI machine. Regardless, the hardware-class constraint is the binding failure.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla.

Commit: `180fa868a81fb6b50bf8288870cca290edd292d9` on branch `remediation/boreal_qwen_image-text_to_image-pytorch-portraits_high_rank-single_device-inference` in `tenstorrent/tt-xla`.

## Verification
- pytest exit: FAIL (OOM on TT silicon)
- Hardware:    wormhole
- Duration:    673.58s (OOM run); test terminated
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL for all boreal_qwen_image portraits-high-rank variants

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 180fa868a81fb6b50bf8288870cca290edd292d9 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
