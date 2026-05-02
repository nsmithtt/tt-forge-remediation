# Remediation Summary: mixtral_8x7b_moe_rp_story_gguf-causal_lm-pytorch-8x7B_MoE_RP_Story_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mixtral_8x7b_moe_rp_story_gguf/causal_lm/pytorch-8x7B_MoE_RP_Story_GGUF-single_device-inference]

## Result
XFAIL — Mixtral 8x7B BF16 (~93 GB) exceeds single-device DRAM (34.2 GB on p150b); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-mixtral-8x7b-bf16-exceeds-p150b-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: TT_FATAL @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 3758096384 B DRAM buffer across 8 banks, where each bank needs to store 469762048 B, but bank size is 4273390016 B (allocated: 3566221504 B, free: 707168512 B, largest free block: 234881024 B)

## Root cause
The Mixtral 8x7B model has 46.7B parameters. At BF16 (2 bytes/param) the full model is ~93 GB, which is 2.7× the p150b device DRAM capacity of 34.2 GB (8 banks × ~4.27 GB). The failing allocation is for `gate_up_proj` tensor of shape [E=8, 2I=28672, H=4096] in BF16 = 3,758,096,384 bytes (3.5 GB) per layer. With 32 layers, the MoE gate_up weights alone require 112 GB — more than 3× the available DRAM. The error fires at `prepareInputTensor` / `ttnn::tilize` when the runtime attempts to transfer the first expert weight tensor to device for execution.

Two loader bugs were fixed during investigation (not related to the OOM):
1. The GGUF file declares `general.architecture = "llama"`, causing the default transformers loader to instantiate `LlamaForCausalLM` and leave all MoE weights randomly initialized, producing PCC 0.29. Fixed by a custom GGUF loader that reads tensors directly and builds the correct `MixtralForCausalLM` state dict with batched expert weights (`gate_up_proj [E,2I,H]`, `down_proj [E,H,I]`).
2. The default `MixtralExperts.forward` dispatches to `batched_mm` (60 GB intermediate) or `grouped_mm` (uses `torch.histc` on int + `_grouped_mm` with no TT lowering). Fixed by a device-friendly forward using `torch.einsum` to avoid materializing permuted weight copies.

Despite both loader fixes, the model cannot run on a single p150b due to hardware capacity.

## Fix
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` entry for this test with OOM reason
- `tt-xla`: `python_package/tt_torch/torch_overrides.py` — added `_mixtral_experts_forward` monkey-patch using `torch.einsum` for device path and per-expert loop for CPU reference path; monkey-patches `MixtralExperts.forward`
- `tt-forge-models`: `mixtral_8x7b_moe_rp_story_gguf/causal_lm/pytorch/loader.py` — complete rewrite of GGUF loading: custom `_gguf_to_mixtral_name`, `_gguf_to_expert_info`, `_load_mixtral_from_gguf` that builds correct transformers 5.x state dict with batched expert tensors; added `requirements.txt` with `gguf>=0.10.0`

## Verification
- pytest exit: FAIL (OOM on device)
- Hardware:    blackhole-p150b
- Duration:    1338.49s (0:22:18)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- tt-xla: `python_package/tt_torch/torch_overrides.py`
- tt-forge-models: `mixtral_8x7b_moe_rp_story_gguf/causal_lm/pytorch/loader.py`
- tt-forge-models: `mixtral_8x7b_moe_rp_story_gguf/causal_lm/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fbda104a8b5985524ce8e229b9c3a22ca4c40ea4 |
| tt-forge-models | 45a0d08087068dbbce2f91db5c0e6af285571fce |
