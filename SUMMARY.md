# Remediation Summary: internlm/causal_lm/pytorch-katuni4ka_tiny_random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[internlm/causal_lm/pytorch-katuni4ka_tiny_random-single_device-inference]

## Result
SILICON_PASS — loader padding clamped to model's max_position_embeddings

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
internlm-rope-position-ids-exceed-max-position-embeddings

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
IndexError: index 128 is out of bounds for dimension 0 with size 128

The error occurs in `apply_rotary_pos_emb` in InternLM's custom modeling code:
```
../../.cache/huggingface/modules/transformers_modules/katuni4ka/tiny_hyphen_random_hyphen_internlm/.../modeling_internlm.py:246: in apply_rotary_pos_emb
    cos = cos[position_ids].unsqueeze(1)
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
IndexError: index 128 is out of bounds for dimension 0 with size 128
```

## Root cause
The `katuni4ka/tiny-random-internlm` model has `max_position_embeddings=128`,
which means `cos_cached` has shape `[128, head_dim]` (indices 0..127).

The loader's `load_inputs` method calls `pad_inputs(inputs["input_ids"], target_len=128)`.
The `pad_inputs` utility adds `max_new_tokens=128` extra padding tokens to the
existing 8-token tokenized input, producing a total padded length of 136.  When
the 136-token sequence is fed to the model, `position_ids` is `arange(0, 136)`,
and `cos[position_ids]` accesses index 128, which is out of bounds.

The loader's `max_length=128` is intended as the number of new tokens to
generate (an inference budget), but `pad_inputs` is additive, so the total
sequence exceeds the model's positional encoding capacity.

## Fix
In `tt_forge_models/internlm/causal_lm/pytorch/loader.py`:

Added `AutoConfig` import and clamped `target_len` before calling `pad_inputs`
so that `seq_len + target_len <= max_position_embeddings`.  After loading the
tokenized inputs (seq_len=8), the fix queries the model config for
`max_position_embeddings=128` and caps `target_len` to `128 - 8 = 120`,
producing a 128-token padded input that fits within the model's positional
encoding table.

File changed: `internlm/causal_lm/pytorch/loader.py` in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    43.02s
- Tier A attempts: N/A

## Files changed
- tt-forge-models: `internlm/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 34c51557e9d556ded37fd7ea2084b3a107e1d2fb |
| tt-forge-models | 349f737231c05adff14360a562ee759017c2d88f |
