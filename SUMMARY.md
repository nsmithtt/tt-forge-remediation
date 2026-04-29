# Remediation Summary: canary-speech_recognition-pytorch-1B_Flash-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[canary/speech_recognition/pytorch-1B Flash-single_device-inference]

## Result
FAIL — after loader fix, compilation hits RuntimeError: Input tensor is not an XLA tensor: XLAComplexFloatType; TT XLA backend does not support complex tensors produced by torch.stft

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
tt-xla-complex-tensor-stft-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: `ERROR: file or directory not found: Flash-single_device-inference]`
(pytest collection failure; space in variant name "1B Flash" splits the test ID on the command line)

After reproducing locally with proper quoting, actual failure:
`ModuleNotFoundError: No module named 'nemo'`

After adding requirements files (loader fix):
```
torch._dynamo.exc.BackendCompilerFailed: backend='tt' raised:
RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLAComplexFloatType

While executing %stft : [num_users=1] = call_function[target=torch.ops.higher_order.wrap_with_autocast](
  args=(xla, None, False, None, %submod_5, %self___featurizer__buffers__window, %masked_fill), kwargs={})
```

## Root cause
Two root causes:

1. **Loader (fixed)**: The canary loader had no `requirements.txt`/`requirements.nodeps.txt`.
   NeMo 2.7.3 requires `torch>=2.6.0` which would replace the custom tt-xla torch, so
   `nemo_toolkit` and torch-dependent packages go in `requirements.nodeps.txt`; all
   other ASR deps (hydra-core, lhotse, nv-one-logger-*, etc.) go in `requirements.txt`.
   Note: `nv-one-logger-core/training-telemetry/pytorch-lightning-integration` are public
   PyPI packages (split distribution), not a monolithic `nv-one-logger`.

2. **Compiler (unfixed, Tier B)**: The Canary model's audio frontend calls `torch.stft`
   (via NeMo's `FilterbankFeatures.forward`), which returns a complex tensor
   (`XLAComplexFloatType`). The TT XLA backend (`tt-xla` PJRT plugin) does not support
   complex-dtype tensors; the check `xtensor:` gate fires immediately when the runtime
   encounters one. Complex tensor support requires new infrastructure in tt-mlir (complex
   TTIR/TTNN ops) and tt-metal (complex dtype kernel support).

## Fix
**Loader fix committed** — added two files to
`canary/speech_recognition/pytorch/`:

- `requirements.txt`: hydra-core, omegaconf, antlr4-python3-runtime, lightning-utilities,
  nv-one-logger-core, nv-one-logger-training-telemetry,
  nv-one-logger-pytorch-lightning-integration, strenum, overrides, toml, wrapt, wget,
  onnx, fiddle, libcst, cloudpickle, lhotse, intervaltree, sortedcontainers, cytoolz,
  toolz, kaldialign, pyannote.core, pyannote.metrics, braceexpand, webdataset, jiwer,
  rapidfuzz, editdistance, inflect, sacremoses, pydub, marshmallow, text-unidecode,
  ruamel.yaml

- `requirements.nodeps.txt`: nemo_toolkit, lightning==2.4.0, pytorch-lightning==2.4.0,
  torchmetrics (installed `--no-deps` to avoid overwriting tt-xla custom torch)

**Compiler bug (not fixed)**: Complex tensor support would need to be added across
tt-mlir (new TTIR/TTNN complex ops) and tt-metal (complex kernel backend), touching
many files across multiple repos — Tier B.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure
Complex tensor support (`XLAComplexFloatType`) requires new TTIR/TTNN ops in tt-mlir
and matching kernel support in tt-metal; this is cross-cutting new infrastructure, not a
scoped one-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    65.81s (model loaded; failed at stft compilation)
- Tier A attempts: N/A

## Files changed
- `canary/speech_recognition/pytorch/requirements.txt` (new)
- `canary/speech_recognition/pytorch/requirements.nodeps.txt` (new)

(in tt-forge-models `remediation/canary-speech_recognition-pytorch-1B_Flash-single_device-inference`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fee12972c3d4115cc684bbc6bfcb7bac40bdfd51 |
| tt-forge-models | cd57c9915110f3cf07b3951e25b49ba0ab9109ab |
