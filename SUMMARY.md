# Remediation Summary: codellama_34b_gguf-causal_lm-pytorch-CodeLlama_34B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[codellama_34b_gguf/causal_lm/pytorch-CodeLlama_34B_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — CodeLlama 34B Q4_K_M (~19 GB weights) exceeds single-device DRAM capacity; OOM during inference tensor transfer

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
oom-codellama-34b-gguf-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Two loader-layer issues were discovered in sequence:

1. The `codellama_34b_gguf` loader lacked a `requirements.txt` specifying `gguf>=0.10.0`. In environments where gguf is not pre-installed or was uninstalled by a prior test's RequirementsManager cleanup, this caused the ImportError reported as the original failure.

2. After applying the `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-26` branch (which includes commit `7aa27e08fd` fixing `_patched_load_gguf_checkpoint` signature compatibility with transformers 5.2.0), the test proceeds past loading and hits the underlying hardware capacity limit.

The fundamental failure is hardware class: CodeLlama 34B Q4_K_M requires approximately 19 GB of DRAM for model weights alone (34B × 0.55 bytes/param). During inference on a single device, the model consumes ~30.9 GB across 8 DRAM banks (3.87 GB/bank), leaving only 216 MB free — insufficient for the additional 360 MB needed for input tensor transfer, causing an OOM in `FlatbufferLoadedExecutableInstance::prepareInputTensor`.

## Fix
- `codellama_34b_gguf/causal_lm/pytorch/requirements.txt` added (gguf>=0.10.0) in `tt-forge-models` on branch `remediation/codellama_34b_gguf-causal_lm-pytorch-CodeLlama_34B_Q4_K_M_GGUF-single_device-inference`.
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` updated with `KNOWN_FAILURE_XFAIL` in `tt-xla` on branch `remediation/codellama_34b_gguf-causal_lm-pytorch-CodeLlama_34B_Q4_K_M_GGUF-single_device-inference`.

## Verification
- pytest exit: XFAIL (1 xfailed, 5 warnings in 227.54s)
- Hardware:    wormhole
- Duration:    227.54s (0:03:47)
- Tier A attempts: N/A

## Files changed
- `codellama_34b_gguf/causal_lm/pytorch/requirements.txt` (new, in tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (modified, in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 775da41cd1d80d5157d302c44b77a36ed6e18878 |
| tt-forge-models | bfe44a6a3d621aecf7a03040e496cb45533a5d75 |
