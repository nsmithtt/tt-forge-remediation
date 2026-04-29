# Remediation Summary: gemma3n_e4b_it_fp8_dynamic-multimodal-pytorch-E4B_IT_FP8_Dynamic-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3n_e4b_it_fp8_dynamic/multimodal/pytorch-E4B_IT_FP8_Dynamic-single_device-inference]

## Result
FAIL — dynamic-shape boolean index in get_placeholder_mask not supported by TT XLA compilation path

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
dynamic-shape-boolean-index-embedding-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Full traceback root:
```
transformers/models/gemma3n/modeling_gemma3n.py:2110: in forward
    special_image_mask, _ = self.get_placeholder_mask(
transformers/models/gemma3n/modeling_gemma3n.py:2005: in get_placeholder_mask
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
torch_xla/_dynamo/dynamo_bridge.py:346: in extract_graph_helper
    torch_xla.sync(reset_scope=False)
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
In `Gemma3nModel.get_placeholder_mask` (transformers/models/gemma3n/modeling_gemma3n.py:2005),
the assertion:

```python
torch_compilable_check(
    inputs_embeds[special_image_mask].numel() == image_features.numel(), ...
)
```

performs `inputs_embeds[special_image_mask]` — a boolean-masked gather whose
output shape depends on how many image tokens appear in the input (data-dependent).
TT's XLA/PJRT compilation path requires fully static tensor shapes. When
`torch_xla.sync()` is called during `extract_compiled_graph`, the device
encounters the dynamic-shape operation and returns INTERNAL: Error code: 13.

This is the same class of failure as the previously documented
`dynamic-shape-boolean-index-embedding-scatter` bug (Qwen3.5/Qwen3VL
get_placeholder_mask), now manifesting in the Gemma3n multimodal model.

A loader fix (adding `compressed-tensors` to requirements.txt) was applied to
reach the compiler failure; the INTERNAL error is the remaining blocker.

## Fix
**Loader fix applied (remediation branch):**
- Added `gemma3n_e4b_it_fp8_dynamic/multimodal/pytorch/requirements.txt` with
  `compressed-tensors` dependency. Without this, `from_pretrained` fails at
  quantization config loading before the model is even constructed.

**Compiler-stack fix needed (not implemented — Tier B):**
The fix must add dynamic-shape tensor support (or a static-shape lowering for
masked boolean gathers) to the TT PJRT/XLA compilation path so that
`inputs_embeds[bool_mask]` can be compiled without producing INTERNAL error 13.

## Tier B justification
**Indicator:** new-infrastructure

Supporting data-dependent tensor shapes in the TT XLA/PJRT compilation
pipeline requires new infrastructure — the static-shape assumption is
pervasive across StableHLO→TTIR lowering and the PJRT execution model. This
is not a scoped one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    349.98s (0:05:49) until INTERNAL error
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: gemma3n_e4b_it_fp8_dynamic/multimodal/pytorch/requirements.txt` (new file, adds compressed-tensors)
- `tt-xla: third_party/tt_forge_models` (submodule pointer updated to remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c9251c8819de914b5c98dac178ac6b9b10f6b74d |
| tt-forge-models | 73a17e5817b6b17c5b256e5e03f92f548dd569a1 |
