# Remediation Summary: enzgamers_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[enzgamers_qwen3_5_35b_a3b_gguf/causal_lm/pytorch-35B_A3B_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — qwen35moe GGUF requires a custom tensor processor for hybrid SSM/MoE architecture; no existing transformers infrastructure handles the mapping

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35moe-ssm-expert-no-tensor-processor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (TypeError — cross-loader contamination):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
After loader fix (KeyError — missing tensor mapping):
```
KeyError: 'blk.0.ffn_gate_exps'
```

## Root cause
Two compounding loader bugs, both in the `loader` layer:

**Bug 1 (fixed):** Cross-loader contamination. Another loader (momix_44) installs a narrow-sig `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` globally during test collection. When transformers 5.2.0+ calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, it hits this narrow patch and raises `TypeError`. The original `_get_real_load_gguf_fn` only searched closure nonlocals for `_orig_load_gguf_checkpoint` — but momix_44 uses `orig_load` as its variable name for the original function. Fix: search both nonlocals and globals, check multiple known variable names (`_orig_load_gguf_checkpoint`, `orig_load`, `original_fn`, `_orig`), and fall back to searching all gguf/checkpoint-named callables.

**Bug 2 (Tier B):** The `qwen35moe` GGUF architecture uses a hybrid SSM+full-attention+MoE layout (GLA layers every 4th full-attention layer). The GGUF file stores expert weight tensors in experts-LAST format (`[hidden=2048, inter=512, experts=256]`), but `Qwen3_5MoeForCausalLM` expects experts-FIRST (`[256, 1024, 2048]` combined `gate_up_proj`). Additionally:
- The GGUF has split `ffn_gate_exps` + `ffn_up_exps` tensors (not combined), and the gguf-py arch mapping incorrectly maps `gate_up_proj → ffn_gate_up_exps` (which does not exist in the GGUF file).
- The `Qwen3_5MoeGatedDeltaNet` SSM parameters (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`) have no mapping to HF parameter names (`A_log`, `in_proj_a`, `in_proj_b`, `conv1d`, `dt_bias`, `norm`, `out_proj`) in any existing `TENSOR_PROCESSORS` entry.
- `Qwen2MoeTensorProcessor` (the closest existing processor) only handles experts-FIRST format and does not perform the axis transposition required here.

## Fix
Bug 1 was fixed in `tt_forge_models` on the `remediation/enzgamers_qwen3_5_35b_a3b_gguf` branch:
- `cbb881084e`: Traverse `orig_load` (and other variable names) when walking the closure chain in `_get_real_load_gguf_fn`
- `ef2f00604e`: Full loader rewrite mapping `qwen35moe` → `qwen3_5_moe_text` with correct `_QWEN35MOE_CONFIG_MAP` (including `expert_feed_forward_length → moe_intermediate_size`), correct `layer_types` list generation from `full_attention_interval`, and patched `get_gguf_hf_weights_map`

**Proposed fix for Bug 2 (not implemented — Tier B):**
A custom `Qwen35MoeTensorProcessor` must be registered in `TENSOR_PROCESSORS["qwen35moe"]` inside `transformers/modeling_gguf_pytorch_utils.py`. It would need to:
1. Transpose expert tensors from `[hidden, inter, experts]` to `[experts, inter, hidden]` before combining
2. Concatenate `ffn_gate_exps` + `ffn_up_exps` into combined `gate_up_proj` with shape `[experts, 2*inter, hidden]`
3. Map SSM parameter names: `ssm_a → A_log`, `ssm_alpha → in_proj_a`, `ssm_beta → in_proj_b`, `ssm_conv1d → conv1d`, `ssm_dt → dt_bias`, `ssm_norm → norm`, `ssm_out → out_proj`

This requires new infrastructure in `transformers` (a new `TensorProcessor` subclass with SSM name mapping + axis transposition + combined tensor construction) and is not a scoped one-function change.

## Tier B justification
new-infrastructure — requires a new `Qwen35MoeTensorProcessor` with SSM weight name mapping, expert tensor axis transposition, and split→combined gate/up projection loading; touches `transformers` internals and cannot be scoped to a single function fix in the loader.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: n/a
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/enzgamers_qwen3_5_35b_a3b_gguf/causal_lm/pytorch/loader.py` (rewritten)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6cf4b45877e19a7f7323a46ed5e522ae84cc5499 |
| tt-forge-models | ef2f00604ea8ee7a71dd917f58ecce6f57d210b5 |
