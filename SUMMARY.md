# Remediation Summary: granite_4_0_h_micro_gguf-causal_lm-pytorch-Granite_4.0_H_Micro_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_4_0_h_micro_gguf/causal_lm/pytorch-Granite_4.0_H_Micro_Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-granitehybrid-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise NotImplementedError(f"Unknown gguf model_type: {model_type} in gguf-py.")
```
Raised in `transformers.modeling_gguf_pytorch_utils.get_gguf_hf_weights_map` because `granitehybrid` was absent from `GGUF_SUPPORTED_ARCHITECTURES` and the associated mapping tables.

## Root cause
The `granite_4_0_h_micro_gguf` loader attempts to load IBM Granite 4.0-H Micro in GGUF format. The GGUF file uses architecture name `granitehybrid`, which maps to HF model_type `granitemoehybrid` (`GraniteMoeHybridForCausalLM`). This architecture was not registered in transformers' GGUF tables (`GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, `GGUF_TO_FAST_CONVERTERS`), causing `get_gguf_hf_weights_map` to raise `NotImplementedError`.

Additionally, the GGUF→HF tensor name mapping required several fixes:
- `attention.head_count_kv` in GGUF is a per-layer array (mostly 0 for Mamba layers); taking `max()` yields the correct `num_key_value_heads` for attention layers.
- `mamba_n_heads` and `mamba_d_head` must be derived from GGUF fields `ssm.time_step_rank` and `ssm.inner_size`.
- Layer types (mamba vs attention) are derived by checking which layers have `blk.N.ssm_in.weight`.
- The `ffn_gate` and `ffn_up` GGUF tensors concatenate into `shared_mlp.input_linear.weight` (gate+up stacking) rather than mapping to separate MoE expert tensors.
- `conv1d.weight` needs an axis inserted ([C, K] → [C, 1, K]) and `A_log`/`D` need squeezing ([N, 1] → [N]).
- `ssm_dt.bias` maps to `mamba.dt_bias`.

## Fix
Single commit in `tt_forge_models` on branch `remediation/granite_4_0_h_micro_gguf-causal_lm-pytorch-Granite_4.0_H_Micro_Q4_K_M-single_device-inference`:

`a12e369609 Fix granite_4_0_h_micro_gguf: register granitehybrid GGUF architecture`

File changed:
- `granite_4_0_h_micro_gguf/causal_lm/pytorch/loader.py`: Added `_patch_transformers_granitehybrid_gguf()` called at import time, which:
  1. Registers `granitehybrid` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING["config"]`.
  2. Registers `GGUFGPTConverter` in `GGUF_TO_FAST_CONVERTERS` for both `granitehybrid` and `granitemoehybrid`.
  3. Installs `_GraniteHybridTensorProcessor` in `TENSOR_PROCESSORS` to handle conv1d shape, A_log/D squeeze, and gate+up concatenation into `input_linear.weight`.
  4. Patches `load_gguf_checkpoint` to remap `model_type` from `granitehybrid` → `granitemoehybrid`, reads extra GGUF fields (SSM params, layer types, scales), and derives `mamba_n_heads`/`mamba_d_head` from `ssm.time_step_rank`/`ssm.inner_size`.
  5. Patches `get_gguf_hf_weights_map` to remap model_type for the weights map and add `ffn_gate/ffn_up → input_linear`, `ffn_down → output_linear`, `ssm_dt.bias → dt_bias` mappings.

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 631.51s (0:10:31)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/granite_4_0_h_micro_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 16c53c7f109dcf4b6180a8773e5c0340903674b2 |
| tt-forge-models | a12e369609c0fdcb10fb8c41146bece690c04793 |
