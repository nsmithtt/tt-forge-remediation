# Remediation Summary: bartowski_thedrummer_valkyrie_49b_v1_gguf-causal_lm-pytorch-49B_V1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_thedrummer_valkyrie_49b_v1_gguf/causal_lm/pytorch-49B_V1_GGUF-single_device-inference]

## Result
XFAIL — Valkyrie 49B Q4_K_M GGUF (~27.6 GB) exceeds P150b single-device DRAM (16 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-model-exceeds-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Loading weights:  49%|████▉     | 280/568 [00:10<00:25, 11.16it/s, Materializing param=model.layers.32.post_attention_layernorm.weight]

## Root cause
The model is TheDrummer_Valkyrie-49B-v1-Q4_K_M.gguf: a 49-billion-parameter model quantized to Q4_K_M (~4.5 bits/param). Total weight storage is approximately 27.6 GB. The test target is a single Blackhole P150b device, which has 16 GB of device DRAM. The model cannot fit on a single P150b regardless of any compiler or runtime change. The CI failure occurs during CPU-side GGUF weight materialization (at 49%, layer 32 of 64) — the exact stop point is consistent with the host runner running out of temporary disk space or being OOM-killed during the materialization step, but even if CPU loading completed, the model would fail to transfer to device DRAM.

Local reproduction was not possible: the NVMe volume at /home/nsmith (/dev/nvme0n1p2, 2.8 TB) was 100% full and prevented downloading the 25+ GB GGUF file. The hardware capacity analysis (27.6 GB > 16 GB) is deterministic and sufficient to classify this as a hardware-class ceiling.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` on remediation branch `remediation/bartowski_thedrummer_valkyrie_49b_v1_gguf-causal_lm-pytorch-49B_V1_GGUF-single_device-inference`.

## Verification
- pytest exit: not-run (disk full; hardware-class determination is sufficient)
- Hardware:    blackhole-p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 27380ffda2f1214092a767d6a5156aa00b5f8f63 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
