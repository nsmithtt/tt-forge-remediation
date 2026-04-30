# Remediation Summary: huihui_qwen_3_coder_next_abliterated-causal_lm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen_3_coder_next_abliterated/causal_lm/pytorch-Huihui_Qwen3_Coder_Next_Abliterated-single_device-inference]

## Result
XFAIL — Qwen3Next 48-layer MoE with 512 experts is ~159 GB BF16, far exceeding n150 DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen3next-moe-512-experts-159gb

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
The model `huihui-ai/Huihui-Qwen3-Coder-Next-abliterated` is a `Qwen3NextForCausalLM` with 48 layers, 512 MoE experts, and hidden_size=2048. The safetensors index reports a total weight size of 159,348,782,592 bytes (~159 GB BF16). The n150 has 12 GB of DRAM. The test timed out because loading ~159 GB of model weights into CPU RAM (let alone GPU DRAM) vastly exceeds the CI timeout. This is a pure hardware capacity ceiling — no compiler or loader bug is involved.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in the tt-xla repo on branch `remediation/huihui_qwen_3_coder_next_abliterated-causal_lm-pytorch-single_device-inference`.

## Verification
- pytest exit: TIMEOUT (reproduced — `timeout 300` killed the test at 5 min; model is 159 GB)
- Hardware:    n150
- Duration:    not-run (XFAIL, no silicon pass attempted)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 951c99a0c2b2daf71214751eeb154a2d7282aa3a |
| tt-forge-models | 7ab35c1b4e023203f9b5e888c083c21bf4f72725 |
