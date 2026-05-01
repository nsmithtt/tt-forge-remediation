# Remediation Summary: mistral-small-3-2-awq-sym-split-sizes-cpu

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/mistral_small_3_2/pytorch-jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym-single_device-inference]

## Result
FAIL — Loader fix for split_sizes CPU computation applied (split_with_sizes bug resolved); blocked by Tier B pjrt-device-to-host-transfer Error code: 13 in comparison phase

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: split_with_sizes expects split_sizes to sum exactly to 2310 (input tensor's size at dimension 0), but got split_sizes=[2320]

(After loader fix) E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause

**Loader bug (fixed):** The `Mistral3PreTrainedModel.get_image_features` computes `split_sizes` using:
```python
split_sizes = (
    (torch.as_tensor(image_sizes, device=image_features.device) // downsample_ratio).prod(dim=-1).tolist()
)
```
When `image_features` is on the XLA/TT device, `image_sizes` (int64) is moved there too. On TT, int64 arithmetic is promoted to bfloat16 internally. The value `42 * 55 = 2310` is not exactly representable in bfloat16 — `bfloat16(2310) = 2320`. So `prod()` returns 2320 instead of 2310, causing `torch.split` to fail.

The fix: patch `get_image_features` on the `Mistral3Model` instance to compute `split_sizes` on CPU via `image_sizes.cpu()` before the integer arithmetic, bypassing TT bfloat16 promotion.

Also needed: `compressed-tensors` package installed (the AWQ checkpoint uses compressed-tensors quantization format, not the traditional autoawq format).

**Tier B bug (unfixed):** After the loader fix, the test reaches the comparison phase and fails with `INTERNAL: Error code: 13` (pjrt-device-to-host-transfer). This occurs when the framework tries to transfer TT device tensors to CPU for PCC comparison. This is a known Tier B bug requiring new PJRT infrastructure for device-to-host tensor transfer.

## Fix
**Loader fix applied** in `tt_forge_models/mistral/mistral_small_3_2/pytorch/loader.py`:
1. Added `compressed-tensors` to `mistral/mistral_small_3_2/pytorch/requirements.txt` so the AWQ checkpoint can be loaded.
2. Added `_patch_get_image_features_cpu_split()` function that overrides `Mistral3Model.get_image_features` as an instance method to compute `split_sizes` on CPU. Key change: `image_sizes.cpu().to(torch.int64) // downsample_ratio` instead of `torch.as_tensor(image_sizes, device=image_features.device) // downsample_ratio`.

**Proposed fix for Tier B:** The `pjrt-device-to-host-transfer` bug requires implementing PJRT tensor transfer paths for TT hardware — new infrastructure work, not a scoped fix.

## Tier B justification
Which indicator: **new-infrastructure**  
The pjrt-device-to-host-transfer error (INTERNAL Error code: 13) is a known Tier B bug throughout the test suite. Fixing it requires implementing new PJRT device-to-host transfer paths for TT hardware, which is a cross-cutting infrastructure change, not a scoped single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    288.28s (0:04:48) for the failing run after loader fix
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mistral/mistral_small_3_2/pytorch/loader.py` — added compressed-tensors dep, split_sizes CPU patch
- `tt_forge_models/mistral/mistral_small_3_2/pytorch/requirements.txt` — created with `compressed-tensors`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 94362e631 |
| tt-forge-models | 797c0a8298 |
