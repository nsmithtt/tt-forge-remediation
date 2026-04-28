# Remediation Summary: armand0e_qwen3_27b_minimax_coder_gguf-causal_lm-pytorch-27B_MiniMax_Coder_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[armand0e_qwen3_27b_minimax_coder_gguf/causal_lm/pytorch-27B_MiniMax_Coder_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 27B model fills ~97% of Blackhole device DRAM, leaving insufficient room for inference activation buffers

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
model-27b-dram-capacity-exceeded

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The CI reported:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After installing gguf (already in requirements-dev.txt as `gguf>=0.10.0`) and using the correct tt-forge-models branch, the test progresses to silicon and hits:

```
RuntimeError: TT_FATAL @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 178257920 B DRAM buffer across 8 banks, where each bank needs to store 22282240 B, but bank size is 4273390016 B (allocated: 4203409728 B, free: 69980288 B, largest free block: 17365440 B)
```

## Root cause
Two layered issues:

1. **Loader (masked by CI venv state)**: The CI ran before the `gguf>=0.10.0` entry in `requirements-dev.txt` was installed. The package was already added (tt-xla commit `cd8104788`) but the venv was stale. Once gguf is installed, this error disappears.

2. **Hardware capacity (the real failure)**: The armand0e Qwen3-27B MiniMax Coder model in Q4_K_M GGUF format, when loaded via `AutoModelForCausalLM.from_pretrained` with `torch_dtype=bfloat16` and `gguf_file`, allocates ~31.4 GB on the Blackhole device (8 banks × 3.92 GB each). The device has only 34.2 GB total DRAM (8 banks × 4.27 GB each). This leaves only ~66 MB free per bank. When executing, the runtime tries to allocate a 22.3 MB tilize output buffer per bank (178 MB total across 8 banks) and finds insufficient contiguous free space (largest free block: 17.4 MB per bank). Inference cannot proceed.

## Fix
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla was updated to mark the test `KNOWN_FAILURE_XFAIL` with a descriptive reason. The tt-forge-models submodule pointer in the remediation branch was also updated to the hf-bringup-21 branch tip (`e52ad04838`) which contains the `model_to_load` kwarg fix for GGUF loaders.

## Verification
- pytest exit: FAIL (TT_FATAL DRAM OOM before xfail config was applied)
- Hardware:    blackhole
- Duration:    848.24s (0:14:08)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` entry for armand0e 27B model
- tt-xla: `third_party/tt_forge_models` — submodule pointer updated to `e52ad04838` (hf-bringup-21 branch tip)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4eab2947f0b0b11da8d661612c38f45e782baeb1 |
| tt-forge-models | e52ad04838f185f07ebba5ffb9e692644148e57a |
