# Remediation Summary: minicpm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[minicpm/pytorch-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-meta-device-rope-buffer-uninitialized

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.

## Root cause
Four compounding transformers 5.x compatibility issues in the loader, all in the `minicpm/pytorch` loader (`tt_forge_models`):

1. **rope_scaling format change**: transformers 5.x normalizes `rope_scaling: null` in `config.json` to `{'rope_type': 'default', 'rope_theta': 10000.0}`. The custom model code (loaded via `trust_remote_code=True`) expects either `None` or a dict with a `'type'` key, causing `KeyError: 'type'` in `_init_rope`.

2. **_tied_weights_keys format change**: transformers 5.x `get_expanded_tied_weights_keys` expects `_tied_weights_keys` to be a dict (`{target: source}`), but the custom model code defines it as a list (`["lm_head.weight"]`), causing `AttributeError: 'list' object has no attribute 'keys'` in `post_init`.

3. **lm_head not in checkpoint**: The checkpoint only stores `model.embed_tokens.weight`; `lm_head.weight` is tied to it. When `tie_word_embeddings=False` is set to skip issue #2, the lm_head gets a random initialization, causing garbage outputs. Manual re-tying after load is required.

4. **Meta-device RoPE buffer corruption (root cause of pcc=nan)**: transformers 5.x uses meta device during `from_pretrained` for efficient loading. Non-persistent buffers (`inv_freq`, `cos_cached`, `sin_cached`) are computed during `__init__` on the meta device (no actual data), then materialized with uninitialized memory when the model is transferred to CPU. The result is garbage RoPE frequency values (e.g., `[3.7451e-25, 0, 0, 0, 1.5695e-43]` instead of `[1.0, 0.75, 0.56, ...]`), causing NaN in all attention outputs.

## Fix
All fixes in `tt_forge_models/minicpm/pytorch/loader.py`:

1. Always load `AutoConfig` before `from_pretrained` and translate `rope_scaling`: if `rope_type='default'`, reset to `None`; otherwise copy `rope_type` to `type`.

2. Set `config.tie_word_embeddings = False` to skip the `_tied_weights_keys` dict-format check in transformers 5.x `post_init`.

3. Set `config.use_cache = False` on the config (not as a kwarg, since the custom model `__init__` doesn't accept it).

4. Set `tokenizer.pad_token = tokenizer.eos_token` when pad_token is missing.

5. After model load, manually re-tie `model.lm_head.weight = model.model.embed_tokens.weight`.

6. After model load, walk all modules and reinitialize `inv_freq`, `cos_cached`, and `sin_cached` for any RoPE module where the buffers are stale from meta-device initialization.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    164.78s (0:02:44)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/minicpm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 41c07ab5be0fa092968dca524187a8f1dbc5f63b |
| tt-forge-models | 97e31d86c66133f06290b57a57b3106873dc6b06 |
