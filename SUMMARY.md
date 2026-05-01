# Remediation Summary: glm_4_7_flash_claude_opus_distill_gguf-causal_lm-pytorch-aiworksofbt_4_7_Flash_Claude_Opus_4_5_High_Reasoning_Distill_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_claude_opus_distill_gguf/causal_lm/pytorch-aiworksofbt_4_7_Flash_Claude_Opus_4_5_High_Reasoning_Distill_GGUF-single_device-inference]

## Result
XFAIL — GLM-4.7-Flash architecture: 64 routed experts, 47 layers, ~32B params (~64 GB BF16 after GGUF dequantization) exceeds single-device DRAM on all supported hardware (12 GB n150, 24 GB p150b)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-7-flash-claude-opus-distill-gguf-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture deepseek2 is not supported yet.
```
from `load_gguf_checkpoint` in transformers 5.2.0 when `AutoConfig.from_pretrained` /
`AutoModelForCausalLM.from_pretrained` is called with `gguf_file` pointing to
`glm-4.7-flash-claude-4.5-opus.q4_k_m.gguf`. The GGUF metadata declares architecture
`deepseek2` which is not registered in `GGUF_CONFIG_MAPPING`.

Cited CI failure: `raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None`
(the Dynamo path on CI may have encountered this differently, but the root failure
is the same hardware capacity ceiling).

## Root cause
**Layer: hardware-class**

The model `aiworksofbt/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF` is a
Claude Opus 4.5 knowledge distillation fine-tune of GLM-4.7-Flash, which uses the
`deepseek2` GGUF architecture (DeepSeek-V2 derived MoE). From the GGUF metadata:

- `deepseek2.block_count: 47` (47 layers)
- `deepseek2.expert_count: 64` (64 routed experts)
- `deepseek2.expert_shared_count: 1` (1 shared expert)
- `deepseek2.embedding_length: 2048` (hidden size)
- `general.size_label: 64x2.6B`

The Q4_K_M GGUF file is 18,132,721,856 bytes (~18.1 GB), corresponding to approximately
32B total parameters at ~4.5 bits/param. After dequantization to BF16 (required for TT
silicon inference), the full weight tensor set occupies ~64 GB.

This exceeds the DRAM capacity of all supported single-device hardware:
- n150: 12 GB DRAM
- p150b: 24 GB DRAM

Additionally, the `deepseek2` architecture is not registered in `GGUF_CONFIG_MAPPING`
in transformers 5.2.0, causing the model to fail at config/weight loading before even
reaching silicon. However, even if all loader bugs were resolved (as documented for the
sibling `mradermacher/GLM-4.7-Flash-heretic-MPOA` model, which required 6 loader fixes),
the hardware capacity ceiling would remain. The sibling variant was separately XFAILed
on the same grounds.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla,
after the existing `glm/causal_lm/pytorch-4.5_Air-single_device-inference` EXCLUDE_MODEL
entry, with a reason string documenting the hardware capacity ceiling.

## Verification
- pytest exit: not-run (hardware-class, model cannot be loaded on single device)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c9533cc4811c21ee2088d540a72deb16082d3cf4 |
| tt-forge-models | a7ad9c1899 |
