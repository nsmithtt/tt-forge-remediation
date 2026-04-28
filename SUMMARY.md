# Remediation Summary: llama4_scout_nvfp4-causal_lm-pytorch-Scout_17B_16E_Instruct_NVFP4-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[llama4_scout_nvfp4/causal_lm/pytorch-Scout_17B_16E_Instruct_NVFP4-single_device-inference]

## Result
XFAIL — Llama-4-Scout-17B-16E MoE model has ~100B total parameters, requiring ~50 GB in NVFP4 (4-bit), which exceeds all single-device DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-100b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception

(When running the current branch the loader fails earlier with: AttributeError: 'Llama4Config' object has no attribute 'pad_token_id', before even reaching the device.)

## Root cause
Llama-4-Scout-17B-16E is a Mixture-of-Experts model with 16 experts per
MoE layer. While only ~17B parameters are active per forward pass (top-1
routing), all expert weights must reside in device memory. The actual
architecture has num_hidden_layers=48, hidden_size=5120,
intermediate_size=8192, and num_local_experts=16, giving approximately
100B total parameters. At NVFP4 (4-bit quantization) that is ~50 GB of
weight storage — far exceeding all single-device DRAM (n150: 12 GB,
p150b: 24 GB).

Additionally, the existing loader used a forbidden workaround: it called
`AutoModelForCausalLM.from_config` with trimmed dimensions
(num_hidden_layers=6, hidden_size=1024) to create a small random-init
model that fits on device. This masked the hardware-capacity limit.

The original `buffer_instance.cc:282` error ("Complex tensor with
num_dims == 0 is not supported") would occur when the trimmed model
reached silicon, because Llama-4's RoPE implementation uses
`torch.view_as_complex` and `torch.polar`, creating complex-typed tensors
in the XLA computation graph that hit the TT PJRT buffer allocation guard.
However, this secondary bug is irrelevant — the model cannot fit on any
single TT device in its full form.

## Fix
Added KNOWN_FAILURE_XFAIL entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
in tt-xla on branch
`remediation/llama4_scout_nvfp4-causal_lm-pytorch-Scout_17B_16E_Instruct_NVFP4-single_device-inference`.

## Verification
- pytest exit: PASS (xfailed, 1 xfailed in 21.46s)
- Hardware:    wormhole
- Duration:    21.46s
- Tier A attempts: N/A

## Files changed
- tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a7de0adf24a131544c93ca3d7dba31b39fd5afcc |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
