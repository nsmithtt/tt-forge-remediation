# Remediation Summary: leo_hessianai_7b-causal_lm-pytorch-leo-hessianai-7b-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[leo_hessianai_7b/causal_lm/pytorch-leo-hessianai-7b-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
tokenizer-padding-max-length-pcc-drop

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.77283285059949. Required: pcc=0.95.

## Root cause
The loader passed `padding="max_length"` with `max_length=128` to the tokenizer. The sample text ("Wie hoch ist der Berg Zugspitze?") tokenizes to only 11 tokens, resulting in 117 padding tokens appended. The TT compiler mishandles the resulting 4D attention mask that combines the padding attention_mask with the causal mask, producing incorrect attention outputs at real token positions. On CPU, padded and unpadded inputs yield PCC=0.999999, confirming the attention mask is handled correctly there but not on device.

## Fix
Removed `padding="max_length"`, `truncation=True`, and `max_length=max_length` from `load_inputs()` in `tt_forge_models/leo_hessianai_7b/causal_lm/pytorch/loader.py`. The tokenizer now returns the natural token sequence (11 tokens) without padding, matching the pattern used by other LLaMA-based loaders (e.g., llama). The unused `max_length` local variable was also removed.

Remediation branch: `remediation/leo_hessianai_7b-causal_lm-pytorch-leo-hessianai-7b-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    149.06s (0:02:29)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/leo_hessianai_7b/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 145504c115783dd2a44fcad5a2bf166ad319cb2f |
