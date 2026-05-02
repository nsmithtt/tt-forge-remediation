# Remediation Summary: nucleotide_transformer-masked_lm-pytorch-InstaDeepAI-nucleotide-transformer-v2-50m-multi-species-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nucleotide_transformer/masked_lm/pytorch-InstaDeepAI/nucleotide-transformer-v2-50m-multi-species-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
transformers-5x-nucleotide-transformer-custom-esm-compat, torch-overrides-mixed-dtype-matmul-linear

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The test was not parametrized (collected 0 items) because `ModelVariant.V2_50M_MULTI_SPECIES` was missing from the loader. After adding it, the model failed to load due to four transformers 5.x incompatibilities in the custom `modeling_esm.py` code from the HuggingFace repo. After fixing those, the model ran on TT silicon but failed at inference time with `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16` in `torch_overrides.py:30` (the 4D einsum path).

## Root cause
Two bugs in two layers:

**Loader layer** (`tt_forge_models`): The V2_50M_MULTI_SPECIES variant was missing from `ModelVariant` enum and `_VARIANTS` dict. Additionally, four transformers 5.x breaking changes affected the custom `modeling_esm.py` from the HuggingFace repo:
1. `find_pruneable_heads_and_indices` removed from `transformers.pytorch_utils`
2. `PreTrainedModel.get_head_mask` removed in transformers 5.x
3. `all_tied_weights_keys` not initialized by the custom `EsmForMaskedLM` under meta-device init context
4. `PretrainedConfig` no longer sets `is_decoder`, `chunk_size_feed_forward`, `add_cross_attention` defaults

**tt-xla layer** (`TorchFunctionOverride`): The custom model's `get_extended_attention_mask` returns a float32 mask (via transformers internals); adding it to bfloat16 attention scores promotes them to float32. The `TorchFunctionOverride.__torch_function__` was only guarding the 4D+ path and did not handle the dtype mismatch — both at 4D+ (original 4D einsum) and at 2D/3D ranks (fallback `func(*args)`). Casting `weight` up to float32 (matching the promoted float32 activation) compounded the problem downstream: the bias stays bfloat16, so `torch.addmm(bf16_bias, f32_input, f32_weight.T)` then failed. The correct fix is to cast the float32 activation DOWN to the model's native bfloat16 (weight's dtype), keeping bias, weight, and activations consistent throughout.

## Fix
**Loader fix** (`tt_forge_models`, `nucleotide_transformer/masked_lm/pytorch/loader.py`):
- Added `V2_50M_MULTI_SPECIES` to `ModelVariant` enum and `_VARIANTS` dict
- Added `_patch_transformers_nucleotide_transformer()` that shims: `find_pruneable_heads_and_indices`, `get_head_mask`, `all_tied_weights_keys` initialization, and three missing PretrainedConfig defaults
- Reverted to `AutoModelForMaskedLM.from_pretrained(trust_remote_code=True)` to use the custom SwiGLU architecture (the HF repo's `EsmIntermediate` uses `2*intermediate_size`)

Branch: `remediation/nucleotide_transformer-masked_lm-pytorch-InstaDeepAI-nucleotide-transformer-v2-50m-multi-species-single_device-inference` in `tt_forge_models`
Commit: `74485ab1ac`

**tt-xla fix** (`tt-xla`, `python_package/tt_torch/torch_overrides.py`):
Extended `TorchFunctionOverride.__torch_function__` to handle dtype mismatches at all tensor ranks (not just 4D+). The trigger condition now includes `inp.dtype != weight.dtype` alongside the existing 4D+ shape check. When dtypes differ, `inp` is cast to `weight.dtype` (activation → model's native dtype) before the einsum, so weight, bias, and activations all stay in bfloat16 throughout the forward pass.

Branch: `remediation/nucleotide_transformer-masked_lm-pytorch-InstaDeepAI-nucleotide-transformer-v2-50m-multi-species-single_device-inference` in `tt-xla`
Commit: `d65115a7ba`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    75.93s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/nucleotide_transformer/masked_lm/pytorch/loader.py` (loader fixes)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (dtype cast fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d65115a7ba4dd53c5a2b37cf94d020db7831267d |
| tt-forge-models | 74485ab1ac94d40f19f7fef63c172fefdae8bfd7 |
