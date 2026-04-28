# Remediation Summary: btbtyler09-qwen3-coder-next-gptq-4bit

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[btbtyler09_qwen3_coder_next_gptq_4bit/causal_lm/pytorch-Qwen3-Coder-Next-GPTQ-4bit-single_device-inference]

## Result
XFAIL — Model is 47 GB on disk in GPTQ 4-bit, far exceeding n150 single-device DRAM (12 GB)

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
Fatal Python error: Segmentation fault

(Current environment also exhibits: `NameError: name 'QuantizeConfig' is not defined` at
`optimum/gptq/quantizer.py:168` because `optimum` 2.1.0 requires `gptqmodel` which is not
installed; and before that: `ImportError: Loading a GPTQ quantized model requires optimum`.
These are pre-loading failures on the path to the segfault.)

## Root cause
`btbtyler09/Qwen3-Coder-Next-GPTQ-4bit` is a MoE (Mixture-of-Experts) model with
`num_hidden_layers=48`, `num_experts=512`, `moe_intermediate_size=512`.
The checkpoint is 47 GB on disk in GPTQ 4-bit format (12 safetensor shards).
At a 4-bit → BF16 expansion of 4×, the unquantized model would be approximately 188 GB.
Even the compressed GPTQ form is 4× larger than the 12 GB DRAM available on an n150 device.
The segfault in the original report was the process crashing during GPTQ kernel initialization
(likely exllama CUDA kernels called on non-CUDA TT hardware), but the underlying reason
the test cannot pass is hardware capacity: the model cannot fit on a single TT device.

## Fix
Marked `btbtyler09_qwen3_coder_next_gptq_4bit/causal_lm/pytorch-Qwen3-Coder-Next-GPTQ-4bit-single_device-inference`
as `KNOWN_FAILURE_XFAIL` in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` with reason:
"Model is 47 GB on disk in GPTQ 4-bit; unquantized equivalent exceeds n150 single-device DRAM (12 GB)"

## Verification
- pytest exit: FAIL (pre-loading — reproducing the segfault requires gptqmodel/auto-gptq which would upgrade transformers)
- Hardware:    n150
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c41961235f1227ade4c85ea0ce4e4bdbd4487c16 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
