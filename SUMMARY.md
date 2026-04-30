# Remediation Summary: h_optimus-feature_extraction-pytorch-H-optimus-0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[h_optimus/feature_extraction/pytorch-H-optimus-0-single_device-inference]

## Result
FAIL — HuggingFace gated model access not approved for the test account; no code fix is possible

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
hf-gated-repo-access-not-granted

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
huggingface_hub.errors.GatedRepoError: 403 Client Error.

Cannot access gated repo for url https://huggingface.co/bioptimus/H-optimus-0/resolve/main/config.json.
Access to model bioptimus/H-optimus-0 is restricted and you are not in the authorized list. Visit https://huggingface.co/bioptimus/H-optimus-0 to ask for access.

(The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a stray Python warning printed after pytest summary; the operative error is the GatedRepoError above.)

## Root cause
`bioptimus/H-optimus-0` is a gated repository on HuggingFace Hub with `gated: auto` approval mode. The test account (`nsmithtt`) has not accepted the model's terms of service, so every call to `hf_hub_download` returns HTTP 403. The loader code itself is correct — it calls `timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, ...)`, which internally calls `hf_hub_download` through timm's `_hub.py`. No token or env-var issue: the `HF_TOKEN` env variable is set and the HF API responds successfully to metadata requests, but file downloads are blocked until the account explicitly accepts the gated terms at https://huggingface.co/bioptimus/H-optimus-0.

## Fix
No code change is required in the loader or compiler stack. The fix is operational: the CI/test account must visit https://huggingface.co/bioptimus/H-optimus-0, accept the terms of service, and wait for auto-approval. Once access is granted, `hf_hub_download` will succeed and the test can proceed to compilation.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — this is not a compiler-stack bug.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    21.70s (both runs)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
