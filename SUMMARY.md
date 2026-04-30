# Remediation Summary: huihui_lfm2_24b_a2b_abliterated_i1_gguf-causal_lm-pytorch-24B_A2B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch-24B_A2B_i1_GGUF-single_device-inference]

## Result
XFAIL — LFM2 24B MoE dequantizes to ~48 GB BF16, exceeding n150 DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture lfm2moe is not supported yet.
```
(Original reported failure was `ImportError: Please install torch and gguf>=0.10.0` due to missing requirements.txt; after adding requirements.txt the real error became the architecture registration.)

## Root cause
Two loader bugs stacked on top of a hardware capacity ceiling:

1. **Missing `gguf>=0.10.0` in requirements.txt** — transformers raises `ImportError` when loading a GGUF checkpoint if the `gguf` package is not declared as a dependency.

2. **`lfm2moe` GGUF architecture not registered** — transformers' `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` only contain `lfm2` (the dense variant). The MoE variant GGUF files declare `general.architecture = "lfm2moe"`, which is not recognized. The HF model_type is `lfm2_moe` (`Lfm2MoeConfig`).

3. **Hardware capacity** — LFM2 24B A2B is a 24B-parameter MoE model (40 layers, hidden_size=2048, 64 experts). Dequantized from Q4_K_M to BF16: 24B × 2 bytes ≈ 48 GB, far exceeding n150 DRAM (12 GB).

## Fix
Both loader bugs fixed in `tt-forge-models` on the remediation branch:

- **`huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt`** — added `gguf>=0.10.0`.

- **`huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`** — added `_patch_lfm2moe_support()` which registers the `lfm2moe` GGUF architecture in `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with the correct field mapping (including MoE fields: `expert_count → num_experts`, `expert_used_count → num_experts_per_tok`, `expert_feed_forward_length → moe_intermediate_size`, `leading_dense_block_count → num_dense_layers`). Also wraps `load_gguf_checkpoint` to rename `model_type` from `lfm2moe` to `lfm2_moe` and converts the per-layer `num_key_value_heads` list to a scalar (same pattern as `lfm2` dense handling at `modeling_gguf_pytorch_utils.py:705-716`).

Test config XFAIL entry added in `tt-xla` at:
- **`tests/runner/test_config/torch/test_config_inference_single_device.yaml`**

## Verification
- pytest exit: not-run (XFAIL marked before hardware run)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-forge-models`: `huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (updated)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 2d485e3f46fb6ddb6f4e5a5a2079021baf4c4bd1 |
| tt-forge-models | 50d77713891dec7009644d37f35e1db7cd8cd238 |
