# Remediation Summary: crow_9b_heretic_gguf-causal_lm-pytorch-9B_HERETIC_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[crow_9b_heretic_gguf/causal_lm/pytorch-9B_HERETIC_GGUF-single_device-inference]

## Result
FAIL — PCC=0.790 on TT silicon (required ≥0.99); root cause is a compiler-stack precision bug in the Qwen3.5 GatedDeltaNet SSM path

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure: `ValueError: GGUF model with architecture qwen35 is not supported yet.`

After loader fix: silicon test ran to completion but failed PCC check:
```
PCC comparison failed. Calculated: pcc=0.7900237004924622. Required: pcc=0.99
```

## Root cause

**Loader bugs (fixed):** Two loader bugs prevented the model from loading at all:

1. **`qwen35` GGUF arch not registered**: `GGUF_SUPPORTED_ARCHITECTURES` does not include `qwen35`; `load_gguf_checkpoint` raised `ValueError` immediately. Fix: register `qwen35`, map it to `qwen3_5_text` model_type, build `layer_types` list from `full_attention_interval`, and add a `_Qwen35TensorProcessor` for the SSM-layer tensor transforms.

2. **Broken `_orig_load_gguf_checkpoint` capture (import-order bug)**: `bartowski_coniccat_qwen3_5_27b_writer_gguf` and ~20 other loaders (alphabetically earlier) have a monkey-patched `load_gguf_checkpoint` with an incompatible 2-argument signature `(gguf_path, return_tensors=False)` that rejects the `model_to_load` kwarg added in transformers 5.2.0. When the crow loader ran `from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint as _orig`, it captured the bartowski broken wrapper (which had already replaced the transformers symbol). Fix: replace the direct import with `_find_real_load_gguf_checkpoint()`, a chain-walker that traverses `__globals__['_orig_load_gguf_checkpoint']` links until it finds the function whose `__name__ == 'load_gguf_checkpoint'` and `__module__ == 'transformers.modeling_gguf_pytorch_utils'`.

**Compiler bug (unfixed, Tier B):** After the loader fixes, the model runs on TT silicon but produces PCC=0.790 (required ≥0.99). Precision analysis:

- FP32 CPU baseline: PCC = 1.000
- CPU BF16 reference: PCC = 0.965
- All-BF16 CPU (simulating dropped `.float()` casts): PCC = 0.880
- TT silicon: PCC = 0.790

The GatedDeltaNet forward pass (`modeling_qwen3_5.py` line ~586) contains explicit FP32 upcasts critical to numerical stability:
```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```
Dropping these casts on CPU explains ~8-point PCC drop (1.0→0.88). TT silicon shows an additional ~9-point drop beyond that (0.88→0.79), indicating the compiler introduces further numerical error in the SSM path beyond simply operating in BF16. This is consistent with the known `ttmlir-f32-precision-not-preserved` bug class, but the magnitude of the additional TT-specific degradation suggests fusion or computation-order artifacts in the DeltaNet recurrence.

## Fix

**Loader fix** in `tt-xla/third_party/tt_forge_models`, branch `remediation/crow_9b_heretic_gguf-causal_lm-pytorch-9B_HERETIC_GGUF-single_device-inference`:

- `crow_9b_heretic_gguf/causal_lm/pytorch/loader.py`: Complete rewrite adding:
  - `_find_real_load_gguf_checkpoint()`: chain-walker to skip broken patches from other loaders
  - `_patch_qwen35_support()`: registers `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_FAST_CONVERTERS`
  - `_patched_load_gguf_checkpoint(*args, **kwargs)`: accepts `model_to_load` kwarg; maps `qwen35`→`qwen3_5_text`; builds `layer_types` from `full_attention_interval=4`
  - `_patched_get_gguf_hf_weights_map()`: maps `qwen3_5_text`→`qwen35` for gguf-py lookup
  - `_Qwen35TensorProcessor`: reshapes `ssm_conv1d.weight` from 2D→3D; applies `log(-weights)` to `ssm_a`
  - `load_inputs()`: adds `use_cache=False` to suppress `Qwen3_5DynamicCache` (not a registered PyTree)
  - `load_shard_spec()`: handles mixed architecture (linear-attention layers have `linear_attn`, full-attention layers have `self_attn`)
  - `_install_patches()`: re-applied at `load_model()` time to win against any later-imported broken overrides

**Proposed compiler fix** (Tier B — not attempted): The `ttmlir-f32-precision-not-preserved` bug requires preserving explicit `.float()` upcasts through all lowering passes. This is a cross-cutting change across tt-mlir and potentially tt-xla.

## Tier B justification
`cross-cutting` — preserving FP32 intermediate upcasts requires changes across multiple tt-mlir lowering passes (TTIR→TTNN, at minimum). Additionally, the extra 9-point gap between all-BF16 CPU (PCC=0.880) and TT silicon (PCC=0.790) indicates a second, distinct compiler-side numerical issue in the DeltaNet recurrence path whose root cause has not been identified.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 3632.52s (1:00:32)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/crow_9b_heretic_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d6ac8ece69f709b4fe404288f910e29f9afe250f |
| tt-forge-models | eb84aa7d557ae1d353b49a192843f4ca8edfa0f7 |
