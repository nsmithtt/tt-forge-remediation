# Remediation Summary: deepseek_r1_distill_qwen_25_5b_brainstorm_gguf-causal_lm-pytorch-25.5B_Brainstorm_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_25_5b_brainstorm_gguf/causal_lm/pytorch-25.5B_Brainstorm_Q4_K_M-single_device-inference]

## Result
XFAIL — 25.5B-parameter model in bfloat16 (~51 GB) exceeds single-device DRAM capacity (~12 GB on N150)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-25b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: TT_FATAL @ /home/ttuser/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 141557760 B DRAM buffer across 8 banks, where each bank needs to store 17694720 B, but bank size is 4273390016 B (allocated: 4096418112 B, free: 176971904 B, largest free block: 8851904 B)

Original reported failure (before the run on silicon): raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
The reported ImportError was a secondary symptom caused by pytest-collection-time namespace pollution: other GGUF model loaders (e.g. tvall43_qwen3_5_4b_heretic_v2_i1_gguf) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a signature that lacked `**kwargs`. When transformers 5.2.0 added `model_to_load=dummy_model` to its internal call, the patched function raised TypeError, which transformers then re-raised as an ImportError.

The fix for this signature issue was already present on the hf-bringup-12 branch of tt_forge_models (`d1c687ff59`). The monorepo was pinned to the older commit (`0f7b734348`) which lacked `**kwargs`.

After switching to the hf-bringup-12 branch (the configured branch for this session), the test progressed past loading and ran on silicon. The true failure is a hardware capacity ceiling: the DeepSeek-R1-Distill-Qwen-25.5B model has 25.5B parameters which in bfloat16 requires ~51 GB of DRAM. The TT N150 device has approximately 12 GB of DRAM. After loading and dequantizing the GGUF checkpoint (Q4_K_M, 15 GB on disk, peaking at ~73 GB host RSS during dequantization), the runtime ran out of device DRAM during model compilation.

## Fix
- Checked out tt_forge_models at `d1c687ff59` (hf-bringup-12 branch), which already contains the `**kwargs` fix for all GGUF loader patches — no additional loader code was written.
- Added `KNOWN_FAILURE_XFAIL` entry for this test to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla (commit `2432040c857c0ad4e96c51a50ddde8ba8972faaa`).

## Verification
- pytest exit: FAIL (OOM — hardware capacity)
- Hardware:    n150
- Duration:    287.05s (0:04:47)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2432040c857c0ad4e96c51a50ddde8ba8972faaa |
| tt-forge-models | d1c687ff59bfe408a0e7bedddc5bc94f6e4d97c4 |
