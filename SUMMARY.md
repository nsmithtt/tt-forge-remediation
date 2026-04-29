# Remediation Summary: bartowski-browser-use-bu-30b-a3b-preview-gguf-image-to-text-pytorch-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_browser_use_bu_30b_a3b_preview_gguf/image_to_text/pytorch-browser_use_bu_30b_a3b_preview_gguf-single_device-inference]

## Result
NO_FIX_NEEDED — the TypeError was introduced by commit be7ac5c7a2 and already fixed by commit 93c49df0b4, both of which are in the hf-bringup-16 branch; failure cannot be reproduced on the configured branch

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patch_transformers_qwen3vlmoe_gguf.<locals>.patched_get_gguf_hf_weights_map() takes from 1 to 4 positional arguments but 5 were given

## Root cause
The failure was introduced by commit `be7ac5c7a2` (2026-04-25T13:52), which added the
`_patch_transformers_qwen3vlmoe_gguf()` function to the bartowski_browser_use_bu_30b_a3b_preview_gguf
loader with an incorrect 4-parameter signature for `patched_get_gguf_hf_weights_map`:

```python
def patched_get_gguf_hf_weights_map(
    hf_model, model_type=None, num_layers=None, qual_name=""
):
```

Transformers 5.x calls `get_gguf_hf_weights_map` with 5 positional arguments
(hf_model, processor, model_type, num_layers, qual_name). The patch replaced the
real function with one that only accepts 4, causing the TypeError when transformers
called it during GGUF weight loading.

The fix was applied in commit `93c49df0b4` (2026-04-25T16:54), "Fix bartowski_browser_use
patched_get_gguf_hf_weights_map: restore processor param", which added `processor` back
as the second positional parameter. This commit is already an ancestor of
`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-16` (confirmed via
`git merge-base --is-ancestor`), so the failure cannot be reproduced on that branch.

## Fix
No fix needed. The loader TypeError was already remediated by commit `93c49df0b4`
in tt-forge-models, which is included in the configured branch hf-bringup-16.

## Verification
- pytest exit: not-run
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
None (no fix required)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a80cdca5d5fc472a5f5e8c8425ac7af99cc8cf11 |
