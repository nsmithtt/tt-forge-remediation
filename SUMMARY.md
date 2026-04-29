# Remediation Summary: cosmos_tokenizer-pytorch-CV8x8x8-720p-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cosmos_tokenizer/pytorch-CV8x8x8-720p-single_device-inference]

## Result
FAIL — loader fix applied (added requirements.txt with cosmos-tokenizer); silicon verification blocked because the HF account in this remediation environment has not accepted the terms of use for the gated repo nvidia/Cosmos-Tokenize1-CV8x8x8-720p and autoencoder.jit cannot be downloaded

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
missing-cosmos-tokenizer-requirements

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   torch._dynamo.exc.InternalTorchDynamoError: ModuleNotFoundError: No module named 'cosmos_tokenizer.modules'

## Root cause
The cosmos_tokenizer/pytorch loader loads an NVIDIA TorchScript JIT model (autoencoder.jit) via torch.jit.load(). When torch.compile / dynamo tries to trace the loaded model it encounters code that references cosmos_tokenizer.modules, but the cosmos_tokenizer package is not installed. The loader had no requirements.txt, so the test runner never installed the dependency.

## Fix
Added cosmos_tokenizer/pytorch/requirements.txt to the tt-forge-models repo on branch remediation/cosmos_tokenizer-pytorch-CV8x8x8-720p-single_device-inference containing:

    cosmos-tokenizer @ git+https://github.com/NVIDIA/Cosmos-Tokenizer

This ensures the test runner installs the cosmos_tokenizer package (which provides cosmos_tokenizer.modules) before loading the model.

Local installation of the package was verified: `import cosmos_tokenizer.modules` succeeds after installation.

Silicon verification was blocked: the HF account associated with HF_TOKEN in this environment has read-only metadata access to nvidia/Cosmos-Tokenize1-CV8x8x8-720p (gated: auto) but has not accepted the model terms of use, so hf_hub_download returns 403 and autoencoder.jit cannot be fetched.

## Verification
- pytest exit: FAIL (GatedRepoError 403 blocked model download before tracing could be tested)
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- cosmos_tokenizer/pytorch/requirements.txt (created) in tenstorrent/tt-forge-models

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 419aa088ae4c06612be230d405160520e69a2d13 |
