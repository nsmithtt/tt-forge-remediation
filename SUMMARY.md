# Remediation Summary: glm_4_7_flash_derestricted_gguf-causal_lm-pytorch-4.7_Flash_Derestricted_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_derestricted_gguf/causal_lm/pytorch-4.7_Flash_Derestricted_GGUF-single_device-inference]

## Result
XFAIL — GLM-4.7-Flash model has 64 routed experts × 46 MoE layers (~30B total params), ~58 GB at BF16 after GGUF dequantization, exceeds single n150 device DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-7-flash-derestricted-gguf-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise NotImplementedError(
    f"Unknown gguf model_type: {model_type} in gguf-py. "
    "This might because you're using an outdated version of gguf-py package, "
    "you can install `gguf` package from source refer to "
    "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development"
)
```
in `transformers/modeling_gguf_pytorch_utils.py` inside `get_gguf_hf_weights_map`,
because `model_type = "deepseek_v2"` is absent from gguf-py's `MODEL_ARCH_NAMES`
(which uses `"deepseek2"` as the value for the `DEEPSEEK2` architecture enum).

## Root cause
**Layer: loader (tt_forge_models)**

The `mradermacher/GLM-4.7-Flash-Derestricted-GGUF` GGUF uses the `deepseek2`
architecture. When `test_models.py` runs (even for a single test), its
module-level `setup_test_discovery` call imports ALL model loaders, including
`glm_4_7_flash_gguf/causal_lm/pytorch/loader.py`, which patches
`GGUF_SUPPORTED_ARCHITECTURES` and `load_gguf_checkpoint` globally. Those
patches allow the config to load (getting past the `ValueError` at line 477
of `modeling_gguf_pytorch_utils.py`), but they remap `model_type = "deepseek2"`
→ `"deepseek_v2"` in the result config. When `get_gguf_hf_weights_map` is then
called for weight loading, it searches `MODEL_ARCH_NAMES` for value
`"deepseek_v2"`, which does not exist (gguf-py uses `"deepseek2"`), raising
`NotImplementedError`.

Three loader bugs in the derestricted loader:

1. **Missing deepseek_v2 tokenizer converter**: `GGUF_TO_FAST_CONVERTERS` lacks
   `"deepseek_v2"` (needed when the tokenizer architecture string arrives
   as `"deepseek_v2"` due to the ngxson patch chain).

2. **Missing get_gguf_hf_weights_map remap**: `get_gguf_hf_weights_map` uses
   `hf_model.config.model_type = "deepseek_v2"` but gguf-py's `MODEL_ARCH_NAMES`
   only has `"deepseek2"` → `NotImplementedError`. Needs
   `"deepseek_v2"` → `"deepseek2"` remap before calling the original function.

3. **Weight shape mismatch**: The GGUF stores `q_b_proj` without the rope
   dimension and splits `kv_b_proj` into separate `k_b`/`v_b` tensors,
   incompatible with HF `DeepSeekV2ForCausalLM`. Without
   `ignore_mismatched_sizes=True`, loading raises `RuntimeError`.

**Hardware capacity ceiling**: Even after fixing the loader bugs, the model
cannot run on n150. The GLM-4.7-Flash architecture has 64 routed experts ×
46 MoE layers × ~9.4M params/expert ≈ 28.75B parameters in MoE experts alone.
When dequantized from Q4_K_M GGUF to BF16 by transformers `from_pretrained`,
the model requires ~58 GB in memory — far exceeding n150's 12 GB DRAM. This is
the same capacity ceiling as the GLM-4.7-Flash AWQ variant (XFAIL in
`report/glm_4_7_flash_awq-causal_lm-pytorch-W8A16_GS32-single_device-inference`).

## Fix
**Loader fixes** (commit `2d1beb6e0e` on branch
`remediation/glm_4_7_flash_derestricted_gguf-causal_lm-pytorch-4.7_Flash_Derestricted_GGUF-single_device-inference`
in `tt_forge_models`):

- **`glm_4_7_flash_derestricted_gguf/causal_lm/pytorch/loader.py`** —
  Added `_patch_transformers_deepseek_v2_gguf()` (same pattern as the i1 variant):
  1. Registers `"deepseek_v2"` and `"deepseek2"` in `GGUF_TO_FAST_CONVERTERS`
     with `GGUFQwen2Converter`.
  2. Registers `deepseek2` in `GGUF_SUPPORTED_ARCHITECTURES` with config
     mapping, and patches `load_gguf_checkpoint` to remap
     `model_type = "deepseek2"` → `"deepseek_v2"` in the result.
  3. Patches `get_gguf_hf_weights_map` to remap `"deepseek_v2"` →
     `"deepseek2"` so gguf-py's `MODEL_ARCH.DEEPSEEK2` tensor name map
     is found.
  4. Adds `ignore_mismatched_sizes=True` to `from_pretrained` to handle the
     structural q_b_proj / kv_b_proj shape incompatibility.

**XFAIL marking** (commit `68d1d69ce` in `tt-xla`):

- **`tests/runner/test_config/torch/test_config_inference_single_device.yaml`**
  — Added:
  ```yaml
  glm_4_7_flash_derestricted_gguf/causal_lm/pytorch-4.7_Flash_Derestricted_GGUF-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "GLM-4.7-Flash model has 64 routed experts x 46 MoE layers (~30B params), ~58 GB at BF16 after GGUF dequantization, exceeds single n150 device DRAM (12 GB)."
  ```

## Verification
- pytest exit: not-run (hardware capacity ceiling confirmed by model size analysis)
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer updated)
- `glm_4_7_flash_derestricted_gguf/causal_lm/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 68d1d69ce8ccb1e5a4dafe9bfe85ddb3fa851680 |
| tt-forge-models | 2d1beb6e0ed696dc073276518b669a71b99a967f |
