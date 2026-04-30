# Remediation Summary: glm_4_7_awq-causal_lm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_awq/causal_lm/pytorch-single_device-inference]

## Result
XFAIL — GLM-4.7-AWQ is a 253B-parameter MoE model; AWQ 4-bit checkpoint is ~126 GB, far exceeding n150 DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 22:59:05.703 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device (assert.hpp:104)

## Root cause
The reported TT_FATAL is a well-known transient hardware initialization error: CI's `failure_patterns.yaml` explicitly excludes "connects to a remote mmio device" from the `tt_fatal` failure category, and 80+ tests in `results_main.yaml` record this error but have `status: SILICON_PASS` (they pass on retry after device reset). The error occurs when an Ethernet core left in a bad state by a prior crash connects to a remote MMIO device.

The underlying model cannot run on n150: GLM-4.7-AWQ is a 253B-parameter MoE model (92 layers, 5120 hidden size, 160 routed experts, 8 experts per token). At AWQ 4-bit quantization, the weight checkpoint is ~126 GB — far beyond n150's 12 GB DRAM.

The previous remediation commit on this branch incorrectly marked SILICON_PASS by loading a tiny random model via `from_config` with heavily reduced dimensions (`num_hidden_layers=6`, `hidden_size=1024`, `n_routed_experts=8`) — a forbidden model-trimming workaround. The CI ETH-core error occurred on a subsequent run of that trimmed-model test, not of the real model.

## Fix
- `glm_4_7_awq/causal_lm/pytorch/loader.py` (tt-forge-models, commit fc16ff01cd): removed all dimension trimming; switched from `from_config` to `from_pretrained` so the loader reflects the actual model. The `_tt_static_glm4_moe_forward` static per-expert implementation is retained because the underlying compiler bugs (grouped_mm `torch.histc` on Int, batched_mm L1 CB overflow) are real and will need fixing when the model becomes runnable on larger hardware.
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla, commit 6f0f1a1e9): added `KNOWN_FAILURE_XFAIL` entry for `glm_4_7_awq/causal_lm/pytorch-single_device-inference`.

## Verification
- pytest exit: not-run
- Hardware:    n150
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `glm_4_7_awq/causal_lm/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 6f0f1a1e93c5dc790918ea9361381650a981191f |
| tt-forge-models | fc16ff01cdc2cb57adeeaacfe2f2993fc46842f6 |
