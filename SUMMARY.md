# Remediation Summary: birefnet-pytorch-BiRefNet-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[birefnet/pytorch-BiRefNet-single_device-inference]

## Result
FAIL — torch.ops.torchvision.deform_conv2d has no lowering in the TT compiler stack

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
torchvision-deform-conv2d-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
NotImplementedError: "deformable_im2col" not implemented for 'BFloat16'

While executing %deform_conv2d : [num_users=1] = call_function[target=torch.ops.torchvision.deform_conv2d.default](args = (%relu_16, %mark_argument_attributes_24, %convolution_3, %mul_288, %full_like_152, 1, 1, 0, 0, 1, 1, 1, 1, True), kwargs = {})
```

The originally-reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` was a symptom of pytest failing at conftest import time; the actual errors surfaced progressively as loader fixes were applied.

## Root cause

Three sequential issues were uncovered and the first two fixed in the loader:

**Issue 1 (fixed): missing kornia dependency.** BiRefNet uses `trust_remote_code=True`; the remote `birefnet.py` module requires `kornia` which was absent. Fixed by adding `kornia` to `birefnet/pytorch/requirements.txt`.

**Issue 2 (fixed): spacy namespace shadowing.** `DynamicLoader.setup_models_path` added `models_root` (the `tt_forge_models/` directory) to `sys.path`. This made `tt_forge_models/spacy/` importable as a top-level `spacy` package, shadowing the real spaCy library. `datasets._dill` then failed with `AttributeError: module 'spacy' has no attribute 'Language'` whenever `load_dataset` computed a fingerprint. Fixed in the `remediation/birefnet-pytorch-BiRefNet-single_device-inference` tt-xla branch by removing the redundant `sys.path.insert`.

**Issue 3 (fixed for CPU reference, compiler bug remains): deform_conv2d BF16 on CPU.** BiRefNet's `DeformableConv2d` module calls `torchvision.ops.deform_conv2d`, which does not implement BFloat16 on CPU. After the spacy fix, the CPU reference run hit this error. Fixed in the loader by patching the `deform_conv2d` reference in the birefnet module's namespace after `from_pretrained` to cast BF16 tensors to float32 on CPU only.

**Issue 4 (unfixed, Tier B): no TT lowering for deform_conv2d.** After the CPU reference run succeeded, torch.compile with the TT backend traced the model graph but could not execute `torch.ops.torchvision.deform_conv2d`. The op fell back to eager execution (through `tt_torch/torch_overrides.py`) with BF16 tensors, which fails. `torchvision.deform_conv2d` is a custom CUDA kernel (`deformable_im2col`) with no ATen decomposition into standard PyTorch ops, so there is no existing path to lower it to StableHLO/TTIR.

## Fix

Loader fixes committed:
- `birefnet/pytorch/requirements.txt` — added `kornia` (commit `3473c60f0a` on `worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-range-300-200-0` in tt-forge-models)
- `birefnet/pytorch/loader.py` — added `_patch_deform_conv2d_bf16_cpu()` static method that replaces the birefnet module's `deform_conv2d` reference with a float32-casting wrapper for CPU (commit `f5a62808f0`)

tt-xla fix committed:
- `tests/runner/utils/dynamic_loader.py` — removed `sys.path.insert(0, models_root)` that caused spacy shadowing (commit `3992bc76b` on `remediation/birefnet-pytorch-BiRefNet-single_device-inference`)

Proposed compiler fix (Tier B, not attempted): implement a StableHLO/TTIR lowering for `torch.ops.torchvision.deform_conv2d`, or register a pure-PyTorch decomposition that decomposes the op into `unfold`, bilinear `grid_sample`, and `scatter`-equivalent ops that the TT backend already handles.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

`torchvision.deform_conv2d` is a custom C++/CUDA kernel (`deformable_im2col`) with no existing ATen or PyTorch functional decomposition. Lowering it to StableHLO requires either: (a) a new TTIR primitive, or (b) a pure-PyTorch functional decomposition registered before tracing. Both paths require new infrastructure touching tt-xla and/or tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    321.13s (0:05:21) to reach TT compilation failure
- Tier A attempts: N/A

## Files changed
- `birefnet/pytorch/requirements.txt` (tt-forge-models, new file)
- `birefnet/pytorch/loader.py` (tt-forge-models, deform_conv2d BF16 patch)
- `tests/runner/utils/dynamic_loader.py` (tt-xla, spacy sys.path fix)
- `third_party/tt_forge_models` (tt-xla submodule pointer)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bf7b273c1c34dc052995a140fcd2b41b52ebae58 |
| tt-forge-models | f5a62808f0a7867e9cfff1d9ad40cf7a21498c9b |
