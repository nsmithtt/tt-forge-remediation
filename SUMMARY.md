# Remediation Summary: asr_19m_v2_en-speech_recognition-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[asr_19m_v2_en/speech_recognition/pytorch-Base-single_device-inference]

## Result
FAIL — `.pt2` torch.export artifact compiled with older PyTorch is incompatible with PyTorch 2.9.1; deserialization fails in `torch.export.load()`

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
model-artifact-torch-export-pytorch-version-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported error:
```
E   AttributeError: 'function' object has no attribute 'parameters'
```

Current reproduction (PyTorch 2.9.1):
```
W torch/export/__init__.py:451] Ran into the following error when deserializing: Only Tensors of floating point and complex dtype can require gradients
E   RuntimeError: We ran into an error when deserializing the saved file.
```

The change in error surface between the original report and current reproduction is explained by PyTorch version differences: in older PyTorch the deserialization partially succeeded but returned a callable function instead of an `nn.Module`, causing `.parameters()` to fail in `compute_mask`; in PyTorch 2.9.1 the stricter deserializer rejects the file outright.

## Root cause
The model `abr-ai/asr-19m-v2-en` (at HF snapshot `0e6ca888`) wraps a pre-compiled
`torch.export` artifact (`niagara-19m-batch.en-cpu.pt2`, 265 MB). This artifact was
serialized with an older PyTorch version that allowed integer-dtype tensors to be
marked with `requires_grad=True`. PyTorch 2.9.1 enforces the invariant that only
floating-point/complex tensors may require gradients; deserialization raises
`RuntimeError` before the module can be constructed.

Two loader bugs were also identified and partially fixed:

1. **Broken `to()` override**: The model's custom `to(self, device, *args, **kwargs)`
   only accepts the string literals `"cpu"` or `"cuda"`. It raised `ValueError`
   when the loader called `model.to(torch.bfloat16)` (dtype, not device), and again
   when the test framework called `model.to(torch.device("cpu"))` (device object vs
   string). Fixed by patching the instance `to()` after loading to normalise the
   argument and skip the artifact loading for dtype-only or non-cpu/cuda device calls.

2. **Primary blocker — `.pt2` version incompatibility**: Even after fixing `to()`,
   `torch.export.load("niagara-19m-batch.en-cpu.pt2").module()` raises `RuntimeError`
   because the artifact was compiled with a PyTorch version that predates the
   float-requires-grad validation. The fix requires the upstream model maintainer
   (`abr-ai/asr-19m-v2-en`) to recompile and republish the `.pt2` artifact with
   PyTorch 2.5+ (where this constraint is enforced on export, preventing invalid
   files from being created).

## Fix
**Partial fix committed** (loader layer, tt_forge_models):

File: `asr_19m_v2_en/speech_recognition/pytorch/loader.py`

- Replaced `model.to(dtype_override)` with a monkey-patched `to()` that handles
  dtype arguments (no-op for a parameterless wrapper), normalises `torch.device`
  objects to strings, and skips `_load_exported_model` for non-cpu/cuda devices
  (e.g. XLA/TT device objects from the test framework).

Commit: `f06a003781a23b80466ddd14d6e2ba146d906361` on branch
`remediation/asr_19m_v2_en-speech_recognition-pytorch-Base-single_device-inference`
(local only; push to origin/tt-forge-models requires credentials).

**Remaining blocker — proposed fix**:

The `abr-ai/asr-19m-v2-en` HF model needs its `.pt2` artifacts recompiled with
PyTorch 2.5+ and republished. No change in this repository can fix the serialization
incompatibility. After the model is updated, the loader fix above should allow the
test to proceed to the forward-pass stage where XLA compilation of the underlying
`torch.fx.GraphModule` can be evaluated.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~25s (fails before silicon compilation)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/asr_19m_v2_en/speech_recognition/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | f06a003781a23b80466ddd14d6e2ba146d906361 (local only) |
