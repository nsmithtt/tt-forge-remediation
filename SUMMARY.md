# Remediation Summary: falcon_h1r_7b_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[falcon_h1r_7b_gguf/causal_lm/pytorch-Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-falcon-h1-arch-missing-transformers-5x

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```
Which collapsed from:
```
ValueError: GGUF model with architecture falcon-h1 is not supported yet.
```

## Root cause
The `falcon-h1` GGUF architecture was completely absent from transformers 5.6.2's
GGUF loading infrastructure. Six gaps required bridging:

1. `GGUF_TO_TRANSFORMERS_MAPPING["config"]` had no entry for `"falcon-h1"`, so
   `load_gguf_checkpoint` raised the unsupported-architecture error.
2. `GGUF_CONFIG_DEFAULTS_MAPPING` had no `"falcon-h1"` entry, so the defaults
   for `mamba_rms_norm` and `mamba_norm_before_gate` were taken from
   `FalconH1Config`'s class defaults (`False` and `True` respectively), which
   are the opposite of what the trained Falcon-H1R-7B checkpoint uses.
3. `get_gguf_hf_weights_map` passes `model_type` to gguf-py which uses the
   hyphenated name `"falcon-h1"`, but after `load_gguf_checkpoint` renames
   `"falcon-h1"` → `"falcon_h1"` (underscore) for AutoConfig resolution, the
   weights-map call would fail with `NotImplementedError`.
4. `GGUF_TO_FAST_CONVERTERS` had no entry for `"falcon-h1"` or `"falcon_h1"`,
   so tokenizer loading raised `KeyError`.
5. GGUF tensor shapes differed from what the HF model expects:
   - `ssm_a`: GGUF stores shape `[num_heads, 1]`, HF expects `[num_heads]` (log-negated).
   - `ssm_d`: GGUF stores shape `[num_heads, 1]`, HF expects `[num_heads]`.
   - `ssm_conv1d.weight`: GGUF stores `[out_ch, K]`, HF Conv1d expects `[out_ch, 1, K]`.
6. Two tensor names had no default gguf-py mapping:
   - `blk.N.ssm_dt.bias` (GGUF) → `model.layers.N.mamba.dt_bias` (HF).
   - `blk.N.ffn_norm` (GGUF, no `.weight` suffix) → `model.layers.N.pre_ff_layernorm.weight`.

## Fix
All fixes are in the loader monkey-patch `_patch_transformers_falcon_h1_gguf()` in
`falcon_h1r_7b_gguf/causal_lm/pytorch/loader.py` of the `tt-forge-models` repo:

1. Added `GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"]` with full field map
   including `attention.key_length → head_dim` (non-standard: 128 ≠ hidden/heads)
   and `ssm.time_step_rank → mamba_n_heads`.
2. Added `GGUF_CONFIG_DEFAULTS_MAPPING["falcon-h1"] = {"mamba_rms_norm": True,
   "mamba_norm_before_gate": False}`.
3. Patched `get_gguf_hf_weights_map` to remap `model_type="falcon_h1"` →
   `"falcon-h1"` before calling gguf-py, and to add suffix-less `ffn_norm` keys.
4. Registered `GGUF_TO_FAST_CONVERTERS` for both `"falcon-h1"` and `"falcon_h1"`.
5. Implemented `FalconH1TensorProcessor` (registered in `TENSOR_PROCESSORS`) with:
   - `ssm_conv1d.weight`: `np.expand_dims(..., axis=1)`.
   - `.ssm_a` (endswith): `np.log(-weights)` then `np.squeeze(..., axis=-1)`.
   - `.ssm_d` (endswith): `np.squeeze(..., axis=-1)`.
   - `perform_fallback_tensor_mapping`: maps `blk.N.ssm_dt.bias` → `.mamba.dt_bias`.

Files changed:
- `falcon_h1r_7b_gguf/causal_lm/pytorch/loader.py` in `tt-forge-models`
  (remediation branch `remediation/falcon_h1r_7b_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference`)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    1025.38s (0:17:05)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/falcon_h1r_7b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b65ec6f283cc5e5fa2f0cf47ccb700a8e7fcd71d |
| tt-forge-models | 5a38c52e6ddd62363e5e35377a65e18ecd75c3f2 |
