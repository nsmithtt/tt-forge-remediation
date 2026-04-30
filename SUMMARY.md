# Remediation Summary: gigaam-speech_recognition-pytorch-RNNT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gigaam/speech_recognition/pytorch-RNNT-single_device-inference]

## Result
FAIL — XLA backend does not support complex float tensors produced by STFT (Tier B new-infrastructure)

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
xla-stft-complex-float-type-unsupported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLAComplexFloatType

While executing %stft : [num_users=1] = call_function[target=torch.ops.aten.stft.default](args = (%reshape, 320, 160, 320, %args_1, False, True, True), kwargs = {})
```

## Root cause
GigaAM RNNT's forward path runs `torchaudio.transforms.MelSpectrogram`
inside the model's preprocessor before any encoder layers.
`MelSpectrogram` calls `torch.stft()`, which produces a complex-valued
tensor (`XLAComplexFloatType`).  The TT XLA backend has no support for
complex tensor types: the `xtensor` check in the XLA runtime fires
immediately and the compilation aborts.

Before this compiler bug is reached, five loader-layer issues had to be
fixed:
1. `torchaudio` missing from `requirements.txt`.
2. `libtorchaudio.so` links against `libtorch_cuda.so` (absent in the
   CPU-only TT environment) — stubbed `torchaudio._extension` before
   import so the pure-Python transforms are available.
3. `pyannote.audio` import inside `get_vad_pipeline()` detected by
   `transformers.check_imports` AST scanner — stubbed `pyannote` in
   `sys.modules`.
4. Transformers 5.x `get_init_context()` runs model `__init__` under a
   meta-device context; `MelScale.__init__` calls `melscale_fbanks()`
   which calls `.any()` → `.item()` on a meta tensor — patched
   `get_init_context` to filter out the meta `torch.device`.
5. `GigaAMModel.__init__` never calls `self.post_init()`, which
   transformers 5.x requires before `_finalize_model_loading` —
   patched `_finalize_model_loading` to call `post_init()` when
   `all_tied_weights_keys` is absent.
6. `load_inputs()` missing the required `feature_lengths` second argument.
7. Loading the model with `torch_dtype=bfloat16` casts `MelScale.fb` to
   bfloat16 while STFT always outputs float32, causing a dtype mismatch.
   The GigaAM model has no bfloat16 CPU path (only CUDA autocast for the
   encoder). Fixed by loading as float32 unconditionally.

The remaining failure is the STFT complex tensor issue in the TT XLA backend.

## Fix
Loader fixes committed on
`remediation/gigaam-speech_recognition-pytorch-RNNT-single_device-inference`
in `tt_forge_models`:

- `gigaam/speech_recognition/pytorch/requirements.txt`: added `torchaudio`
- `gigaam/speech_recognition/pytorch/loader.py`: five patches:
  - `_patch_torchaudio()`: stub `torchaudio._extension` before import
  - `_patch_pyannote()`: stub `pyannote` in `sys.modules`
  - Patch `PreTrainedModel.get_init_context` to strip meta device
  - Patch `PreTrainedModel._finalize_model_loading` to call `post_init()`
  - `load_inputs()`: return `[audio_tensor, feature_lengths]`
  - `load_model()`: do not pass `torch_dtype` to `from_pretrained`

The tt-xla compiler-stack bug (`xla-stft-complex-float-type-unsupported`)
is left unfixed. A fix would require implementing complex tensor support in
the TT XLA backend — new infrastructure affecting the XLA tensor type
system, the STFT lowering path, and potentially tt-mlir's StableHLO
handling.

## Tier B justification
The STFT operation produces a complex-valued output
(`XLAComplexFloatType`).  The TT XLA backend has no complex tensor type
at all — the `xtensor` guard fires before any lowering attempt.  Fixing
this requires: adding `XLAComplexFloatType` as a first-class XLA tensor
type, implementing complex tensor arithmetic, and providing a StableHLO
→ TTIR lowering for STFT (or decomposing STFT into real ops before it
reaches the backend).  This is new-infrastructure: it touches the XLA
runtime type system, the dynamo export path, and tt-mlir, spanning more
than three files across at least two repos.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole (n150)
- Duration:    48.88s (wall clock, failed at TT compilation)
- Tier A attempts: N/A

## Files changed
- `gigaam/speech_recognition/pytorch/requirements.txt`
- `gigaam/speech_recognition/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | b7ebbd0db9ddc70f7b3e59e56bc84e48221c81cf   |
| tt-forge-models | 431e021dc895569b80fb8b19e067d2b1db453019   |
