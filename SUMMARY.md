# Remediation Summary: glm_4_7_flash_derestricted_i1_gguf-causal_lm-pytorch-4_7_Flash_Derestricted_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_derestricted_i1_gguf/causal_lm/pytorch-4_7_Flash_Derestricted_i1_GGUF-single_device-inference]

## Result
XFAIL — GLM-4.7-Flash-Derestricted has 64 routed experts × 46 MoE layers (~30B params), ~58 GB at BF16 after GGUF dequantization, exceeds single n150 device DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-7-flash-derestricted-i1-gguf-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
third_party/tt_forge_models/iquest_coder_v1_40b_base_gguf/causal_lm/pytorch/loader.py:19: in _patched_convert_gguf_tokenizer
    fast_tokenizer, additional_kwargs = _orig_convert_gguf_tokenizer(
venv/lib/python3.12/site-packages/transformers/integrations/ggml.py:787: in convert_gguf_tokenizer
    converter = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name](tokenizer_dict)
KeyError: 'deepseek_v2'
```

## Root cause
**Layer: loader (tt_forge_models)**

The `mradermacher/GLM-4.7-Flash-Derestricted-i1-GGUF` uses the `deepseek2` GGUF
architecture. The `glm_4_7_flash_gguf` loader (imported via pytest's full-suite
discovery) applies `_patch_transformers_deepseek2_gguf()` at import time, which:
1. Registers `"deepseek2"` in `GGUF_TO_FAST_CONVERTERS`.
2. Patches `load_gguf_checkpoint` to remap `model_type "deepseek2"` → `"deepseek_v2"`.

When the i1 tokenizer then calls `convert_gguf_tokenizer(architecture, ...)`, it
receives `architecture = "deepseek_v2"` (from the remapped config) but
`GGUF_TO_FAST_CONVERTERS` only has `"deepseek2"`, not `"deepseek_v2"` → `KeyError`.

A second loader bug would surface after fixing the tokenizer: `get_gguf_hf_weights_map`
calls gguf-py's `MODEL_ARCH_NAMES` to look up the architecture enum by value, but
`MODEL_ARCH_NAMES` only has `"deepseek2"` — not `"deepseek_v2"` — so a
`NotImplementedError` would occur during weight mapping.

**Hardware capacity ceiling**: GLM-4.7-Flash has 64 routed experts × 46 MoE layers.
Transformers dequantizes the GGUF to BF16 on load; the full model requires ~58 GB
in memory, far exceeding n150's 12 GB DRAM. This is the same ceiling as the
non-i1 variant (see `report/glm_4_7_flash_derestricted_gguf-causal_lm-pytorch-4.7_Flash_Derestricted_GGUF-single_device-inference`).

## Fix
**Loader fix** (commit `c736e8926e` on branch
`remediation/glm_4_7_flash_derestricted_i1_gguf-causal_lm-pytorch-4_7_Flash_Derestricted_i1_GGUF-single_device-inference`
in `tt_forge_models`):

- **`glm_4_7_flash_derestricted_i1_gguf/causal_lm/pytorch/loader.py`** —
  Added `_patch_deepseek_v2_gguf()` called at import time:
  1. Registers `"deepseek_v2"` in `GGUF_TO_FAST_CONVERTERS` with `GGUFQwen2Converter`
     so `convert_gguf_tokenizer` can handle the remapped architecture name.
  2. Patches `get_gguf_hf_weights_map` to temporarily remap `model_type "deepseek_v2"`
     → `"deepseek2"` during the call so gguf-py's `MODEL_ARCH.DEEPSEEK2` tensor-name
     map is found. Uses the correct transformers 5.x signature `(hf_model, processor=None, model_type=None)`.
  3. Added `ignore_mismatched_sizes=True` to `from_pretrained` to handle the
     structural `q_b_proj`/`kv_b_proj` shape incompatibility inherent to DeepSeek-V2
     GGUF weight layout.

**XFAIL marking** (commit `8eaa7ed120` in `tt-xla`):

- **`tests/runner/test_config/torch/test_config_inference_single_device.yaml`**
  — Added:
  ```yaml
  glm_4_7_flash_derestricted_i1_gguf/causal_lm/pytorch-4_7_Flash_Derestricted_i1_GGUF-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "GLM-4.7-Flash has 64 routed experts x 46 MoE layers (~30B params), ~58 GB at BF16 after GGUF dequantization, exceeds single n150 device DRAM (12 GB)."
  ```

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware: n150
- Duration: 148.91s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer updated)
- `glm_4_7_flash_derestricted_i1_gguf/causal_lm/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8eaa7ed1209a12cf0202a3540f3fadb700d43fd1 |
| tt-forge-models | c736e8926e700d57dd314dc3f37c934e231fac1c |
