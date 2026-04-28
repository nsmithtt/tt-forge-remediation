# Remediation Summary: alvocat_vocos_22khz-pytorch-alvocat_22khz-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[alvocat_vocos_22khz/pytorch-alvocat_22khz-single_device-inference]

## Result
FAIL — `torch.fft.irfft` lowers to `aten._fft_c2r` which has no lowering in the compiler stack (Tier B: new-infrastructure)

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
aten-fft-c2r-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: Error code: 13

While executing %_fft_c2r : [num_users=1] = call_function[target=torch.ops.aten._fft_c2r.default](args = (%mul_29, [1], 2, 1024), kwargs = {})
Original traceback:
  File ".../alvocat_vocos_22khz/pytorch/loader.py", line 40, in forward
    audio = self.head(x)
  File ".../vocos/heads.py", line 68, in forward
    audio = self.istft(S)
  File ".../vocos/spectral_ops.py", line 56, in forward
    ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
```

(The originally reported `TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device` was not reproduced; four loader bugs were fixed first, then a Tier A compiler fix eliminated a 0-dim complex tensor error, revealing this Tier B blocker.)

## Root cause

The `ISTFTHead` in `vocos/spectral_ops.py` calls `torch.fft.irfft`, which lowers to `aten._fft_c2r` in the ATen graph. The tt-xla compiler stack (via StableHLO → tt-mlir) has no lowering for FFT operations. There is no handler for `stablehlo.fft` or `aten._fft_c2r` in tt-mlir's lowering passes, so the op fails at graph execution time with error code 13.

Four loader-layer bugs were fixed en route to uncovering this:
1. Missing `requirements.txt` (no `vocos` or `torchaudio` listed).
2. `torchaudio` from PyPI required CUDA; pinned to `torchaudio==2.9.1+cpu`.
3. `dynamic_loader.py` inserts `models_root` at `sys.path[0]`, causing `tt_forge_models/encodec/` to shadow the `encodec` pip package required by vocos. Fixed by temporarily removing the conflicting path before importing vocos.
4. `vocos 0.1.0`'s `MelSpectrogramFeatures.__init__` does not accept the `f_min`, `f_max`, `norm`, `mel_scale` kwargs present in the alvocat HuggingFace config. Fixed by bypassing `Vocos.from_pretrained()` entirely — instantiating `VocosBackbone` and `ISTFTHead` directly from config and loading state dict entries by prefix.

A Tier A fix was also applied in `tt-xla`: `buffer_instance.cc` previously threw for 0-dim complex tensors (`1j` literal in `vocos/heads.py`). That guard was removed, allowing a 0-dim complex to be represented as a shape-`{2}` float tensor. This fix succeeded (error eliminated, test advanced), but the test then failed on `aten._fft_c2r`.

## Fix

### Loader fixes (tt_forge_models, branch: remediation/alvocat_vocos_22khz-pytorch-alvocat_22khz-single_device-inference)

- `alvocat_vocos_22khz/pytorch/requirements.txt` — created with `vocos>=0.1.0` and `torchaudio==2.9.1` on the CPU index URL.
- `alvocat_vocos_22khz/pytorch/loader.py` — rewrote `load_model()` to: (a) temporarily remove conflicting `encodec`-containing `sys.path` entries before importing `vocos.models` / `vocos.heads`; (b) bypass `Vocos.from_pretrained()` by downloading `config.yaml` via `hf_hub_download`, instantiating backbone and head from their `init_args`, and loading weights from `pytorch_model.bin` by prefix-stripped state dict.

### Tier A fix (tt-xla, branch: remediation/alvocat_vocos_22khz-pytorch-alvocat_22khz-single_device-inference)

- `pjrt_implementation/src/api/buffer_instance.cc` — removed the early-exit `TT_THROW` for 0-dim complex tensors in `calculateShape()`. A 0-dim complex now falls through to the existing `shape.push_back(2)` encoding, producing a shape-`{2}` float tensor for the interleaved real/imag representation.

### Proposed fix for Tier B blocker (`aten._fft_c2r` / `torch.fft.irfft`)

Implement a lowering for `stablehlo.fft` (type IRFFT) in tt-mlir, or add a CPU fallback path in tt-xla for `aten._fft_c2r`. This requires new kernel infrastructure or tt-metal FFT support — it is not a scoped single-file change.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure — `aten._fft_c2r` / `stablehlo.fft` has no lowering anywhere in the tt-mlir / tt-metal stack. Supporting FFT requires new kernel implementations or a significant new lowering pass; it is not a bounded single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    16.36s
- Tier A attempts: 1

## Files changed
- `tt-xla/pjrt_implementation/src/api/buffer_instance.cc` (Tier A fix)
- `tt-xla/third_party/tt_forge_models/alvocat_vocos_22khz/pytorch/loader.py` (loader fix)
- `tt-xla/third_party/tt_forge_models/alvocat_vocos_22khz/pytorch/requirements.txt` (loader fix, new file)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 70176024b728c15059df4c68aa3b9573b12b7c80 |
| tt-forge-models | a6cb5c032dcff3cd48015513bb542acbb8b97290 |
