# Remediation Summary: k2_v2-causal_lm-pytorch-K2_V2_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[k2_v2/causal_lm/pytorch-K2_V2_Instruct-single_device-inference]

## Result
XFAIL — model is a 70.5B-parameter LlamaForCausalLM (~141 GB at BF16), which exceeds single p150b device DRAM capacity (144 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-llama-single-device

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
LLM360/K2-V2-Instruct is a 70.5B-parameter LlamaForCausalLM (80 layers,
hidden_size=8192, intermediate_size=28672, vocab_size=250112). At BF16 the
model weights alone occupy ~141 GB; the single p150b device has 144 GB DRAM.
With no headroom left for KV-cache, activation buffers, or runtime
allocations, the model cannot run on a single device. CI timed out before
or during loading because the weights were not cached on the runner and/or
the device OOM caused an unclean hang rather than a clean failure.

## Fix
Added KNOWN_FAILURE_XFAIL entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
in tt-xla, on branch
`remediation/k2_v2-causal_lm-pytorch-K2_V2_Instruct-single_device-inference`.

## Verification
- pytest exit: TIMEOUT (not-run on silicon — model too large to download/load for re-run)
- Hardware:    blackhole-p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6f4447ef678cb087917f94302816b17586a15f00 |
| tt-forge-models | e8a5fee0d7ed4c15051f7705398c01e72dad729a |
