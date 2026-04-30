# Remediation Summary: huihui_ling_mini_2_0_abliterated_i1_gguf-causal_lm-pytorch-HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_ling_mini_2_0_abliterated_i1_gguf/causal_lm/pytorch-HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF-single_device-inference]

## Result
XFAIL — BailingMoeV2 (~16.5B params, 256 experts × 19 MoE layers) dequantized to BF16 exceeds p150b 12 GB DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-bailingmoe2-16b-exceeds-p150b-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: GGUF model with architecture bailingmoe2 is not supported yet.
(raised from transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint)

## Root cause
Two distinct issues:

1. **Loader bug (fixed):** `transformers 5.x` does not include `bailingmoe2` in
   `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`,
   `GGUF_TO_FAST_CONVERTERS`, or `TENSOR_PROCESSORS`. The loader was missing all
   four GGUF registration patches. Additionally, `BailingMoeV2Config` (loaded via
   remote code from `inclusionAI/Ling-mini-2.0`) has `model_type = ""` (inherited
   from `PretrainedConfig`) rather than `"bailing_moe"`, blocking `AutoConfig.register`.
   Two transformers 5.x breaking changes also needed shimming: `is_torch_fx_available`
   removed from `transformers.utils.import_utils`, and `ROPE_INIT_FUNCTIONS['default']`
   removed from `transformers.modeling_rope_utils`.

2. **Hardware capacity ceiling (XFAIL):** BailingMoeV2 has 256 experts × 19 MoE
   layers plus dense layer 0. Total parameter count ≈ 16.5B; Q4_K_M GGUF
   dequantized to BF16 ≈ 33 GB. The p150b has 12 GB on-device DRAM — the model
   is physically too large to fit on a single device.

## Fix
**Loader fix** (`tt_forge_models`, branch `remediation/huihui_ling_mini_2_0_abliterated_i1_gguf-...`):

- `huihui_ling_mini_2_0_abliterated_i1_gguf/causal_lm/pytorch/loader.py`:
  - Added `BailingMoeV2TensorProcessor` class to split fused GGUF expert tensors
    `blk.N.ffn_{gate,up,down}_exps` (shape `[num_experts, dim1, dim2]`) into
    per-expert HF keys `model.layers.N.mlp.experts.K.{gate,up,down}_proj.weight`.
  - Added `_patch_transformers_bailingmoe2_gguf()` called at import time:
    - Shim `is_torch_fx_available` (removed in transformers 5.x)
    - Register `BailingMoeV2Config` and `BailingMoeV2ForCausalLM` (via
      `get_class_from_dynamic_module`) with `AutoConfig`/`AutoModelForCausalLM`;
      patch `BailingMoeV2Config.model_type = "bailing_moe"` before registering.
    - Register `"bailingmoe2"` in `GGUF_SUPPORTED_ARCHITECTURES`,
      `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, `GGUF_TO_FAST_CONVERTERS`
      (`GGUFQwen2Converter`), and `TENSOR_PROCESSORS`.
    - Patch `load_gguf_checkpoint` to remap `model_type bailingmoe2 → bailing_moe`.
    - Patch `get_gguf_hf_weights_map` to reverse-remap `bailing_moe → bailingmoe2`
      for the gguf arch lookup.
    - Shim `ROPE_INIT_FUNCTIONS['default']` (removed in transformers 5.x).
  - Added `_gguf_load_ctx()` context manager (wraps `from_pretrained`) to handle
    `model_to_load` kwarg injected by transformers 5.x that strict-signature patched
    loaders from other models in the same pytest session would reject.

**XFAIL entry** (`tt-xla`, branch `remediation/huihui_ling_mini_2_0_abliterated_i1_gguf-...`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  Added `KNOWN_FAILURE_XFAIL` entry with explanation of the hardware capacity ceiling.

## Verification
- pytest exit: XFAIL (1 xfailed, 6 warnings in 19.75s)
- Hardware:    not-run (XFAIL before device execution)
- Duration:    19.75s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models/huihui_ling_mini_2_0_abliterated_i1_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e23f1c593a393b885a49bebc0043b9d3c451cacf |
| tt-forge-models | 063b5869b002f3380c8955f90fb79423e865cf01 |
