# Remediation Summary: braindecode_bendr-eeg_classification-pytorch-bendr_pretrained-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[braindecode_bendr/eeg_classification/pytorch-bendr_pretrained-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
braindecode-bendr-missing-requirements-safetensors-dtype

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ModuleNotFoundError: No module named 'braindecode'
```
(surfaced as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
due to missing `source venv/activate` in the initial repro attempt — the real error was the
missing dependency once the correct environment was activated)

Subsequent failures after fixing each bug:
1. `OSError: Could not load this library: _torchaudio.abi3.so` — torchaudio 2.11.0 from PyPI
   is ABI-incompatible with the custom torch 2.9.1+cpu build
2. `huggingface_hub.errors.RemoteEntryNotFoundError: 404` for `pytorch_model.bin` — the HF
   repo only contains `model.safetensors`
3. `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` —
   braindecode's `Contextualizer` creates the start_token embedding via `torch.full()` without
   a `dtype` argument, defaulting to float32; the subsequent `torch.cat` promotes the
   bfloat16 sequence to float32, which then mismatches the bfloat16 transformer weights

## Root cause
Three independent loader bugs in `braindecode_bendr/eeg_classification/pytorch/`:

1. **Missing requirements.txt** — `braindecode` was never declared as a dependency for this
   model loader, so the `RequirementsManager` never installed it.

2. **Wrong torchaudio version** — `braindecode` depends on `torchaudio>=2.0`, but without an
   explicit pin pip resolves to torchaudio 2.11.0 from PyPI, which was compiled against a
   newer torch ABI (`torch_library_impl` symbol) not present in the custom torch 2.9.1+cpu
   build used by this environment. Pinning `torchaudio==2.9.1+cpu` from the pytorch CPU index
   fixes this.

3. **Wrong checkpoint filename** — the loader tried to download `pytorch_model.bin`, but the
   `braindecode/braindecode-bendr` HuggingFace repo only ships `model.safetensors`.

4. **Float32 start_token dtype mismatch** — `braindecode.models.bendr.Contextualizer.forward`
   calls `torch.full((1, seq, dim), float(start_token), device=x.device)` without specifying
   `dtype`, producing a float32 tensor. `torch.cat([token_emb, x], dim=0)` then promotes the
   bfloat16 sequence to float32. The transformer layers receive a float32 input against
   bfloat16 weights, causing the RuntimeError.

## Fix
All changes in `tt-forge-models`, file
`braindecode_bendr/eeg_classification/pytorch/loader.py` and
`braindecode_bendr/eeg_classification/pytorch/requirements.txt`:

- Added `requirements.txt` declaring `braindecode` and pinning
  `torchaudio==2.9.1+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu`.
- Changed `hf_hub_download(filename="pytorch_model.bin")` to `filename="model.safetensors"`
  and replaced `torch.load()` with `safetensors.torch.load_file()`.
- After `model.to(dtype_override)`, registered a forward pre-hook on each
  `contextualizer.transformer_layers` entry that casts the sequence tensor back to
  `dtype_override` before the linear projection, compensating for braindecode's
  untyped `torch.full()` start-token creation.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    51.48s
- Tier A attempts: N/A

## Files changed
- `braindecode_bendr/eeg_classification/pytorch/requirements.txt` (new)
- `braindecode_bendr/eeg_classification/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d3f27bb63263e1b050dcdeffa20964e7c08720e8 |
| tt-forge-models | 8136ecc3fd7d914a3d89efab20eeff684cd84164 |
