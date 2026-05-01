# Remediation Summary: layerd_birefnet-pytorch-layerd-birefnet-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[layerd_birefnet/pytorch-layerd-birefnet-single_device-inference]

## Result
FAIL — INTERNAL: Error code: 13 from `_run_cached_graph`; root cause is `torchvision.ops.deform_conv2d` with no TT lowering

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

at `torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)` in `dynamo_bridge.py:611`, after ~14 minutes of compilation. No device-level error text is emitted; only the PJRT status code is surfaced.

## Root cause

Five sequential loader issues were uncovered and all five fixed:

**Issue 1 (fixed): missing kornia dependency.** `cyberagent/layerd-birefnet` uses `trust_remote_code=True`; the remote `birefnet.py` imports `from kornia.filters import laplacian`, which was absent. Fixed by creating `layerd_birefnet/pytorch/requirements.txt` with `kornia`.

**Issue 2 (fixed): meta-tensor `.item()` crash in SwinTransformer backbone.** `transformers` 5.x instantiates models under `torch.device("meta")`. `SwinTransformer.__init__` calls `torch.linspace(...).item()` to build drop-path rates; meta tensors have no data so `.item()` raises `RuntimeError: item() cannot be called on meta tensors`. Fixed by temporarily patching `torch.Tensor.item` to return `0.0` for meta-device tensors during `from_pretrained`.

**Issue 3 (fixed): missing `all_tied_weights_keys` (post_init not called).** `transformers` 5.x `_finalize_model_loading` requires `all_tied_weights_keys` which `post_init()` sets. `BiRefNet.__init__` does not call `post_init()`. Fixed by patching `PreTrainedModel._finalize_model_loading` to call `model.post_init()` when the attribute is absent.

**Issue 4 (fixed for CPU reference, compiler bug remains): deform_conv2d BF16 on CPU.** `DeformableConv2d` calls `torchvision.ops.deform_conv2d` which does not support BFloat16 on CPU (`deformable_im2col not implemented for 'BFloat16'`). Fixed by monkey-patching `DeformableConv2d.forward` to cast inputs to float32 before the deform_conv2d call (CPU only), then cast the output back.

**Issue 5 (fixed): `load_dataset` spacy namespace collision.** The loader called `load_dataset("huggingface/cats-image")` which triggers `datasets._dill` fingerprinting, which in turn imports `spacy`. If `tt_forge_models/spacy/` is on `sys.path` it shadows the real spaCy package. Fixed by replacing `load_dataset` with `PIL.Image.new("RGB", (1024, 1024))` (synthetic input).

**Issue 6 (unfixed, Tier B): no TT lowering for `deform_conv2d`.** After all loader fixes, the CPU reference run succeeds. torch.compile traces the model graph (including the patched `DeformableConv2d.forward` which calls `torchvision.ops.deform_conv2d` with float32 inputs). The TT backend compiles the graph (~14 min) and returns a `graph_hash`, but `_run_cached_graph` raises `INTERNAL: Error code: 13` with no further diagnostic. The probable cause is `torch.ops.torchvision.deform_conv2d` reaching the TT device runtime with no lowering; the exact failure mechanism (incorrect lowering, OOM from fallback allocation, or PJRT D2H transfer) is not captured in the available logs.

## Fix

Loader fixes committed (all in `layerd_birefnet/pytorch/loader.py` and `requirements.txt` on `remediation/layerd_birefnet-pytorch-layerd-birefnet-single_device-inference` in tt-forge-models):
- `bc44800d80` — add `kornia` to `layerd_birefnet/pytorch/requirements.txt`
- `bbfc27853f` — patch `torch.Tensor.item` to return `0.0` for meta tensors during `from_pretrained`
- `e3294882c9` — patch `PreTrainedModel._finalize_model_loading` to call `post_init()` when `all_tied_weights_keys` is absent
- `ebf383ab5e` — patch `DeformableConv2d.forward` to float32-cast deform_conv2d call for CPU BF16 compat
- `eb93515304` — replace `load_dataset` with `PIL.Image.new` to avoid spacy namespace collision

Proposed compiler fix (Tier B, not attempted): implement a StableHLO/TTIR lowering for `torch.ops.torchvision.deform_conv2d`, or register a pure-PyTorch decomposition (via `unfold`, bilinear `grid_sample`, and scatter equivalents) before tracing. Both approaches require new infrastructure in tt-xla and/or tt-mlir.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

`torchvision.deform_conv2d` is a custom C++/CUDA kernel (`deformable_im2col`) with no existing ATen or PyTorch functional decomposition into standard ops. Lowering it to StableHLO/TTIR requires either a new TTIR primitive or a registered pure-PyTorch decomposition — both require new infrastructure. Additionally, the exact runtime failure mechanism (INTERNAL error 13) is not captured, satisfying the `internal-error-unknown-mechanism` indicator as well.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    842.68s (0:14:02)
- Tier A attempts: N/A

## Files changed
- `layerd_birefnet/pytorch/requirements.txt` (tt-forge-models, new file — kornia dep)
- `layerd_birefnet/pytorch/loader.py` (tt-forge-models — 4 loader fixes)
- `third_party/tt_forge_models` (tt-xla submodule pointer → eb93515304)
- `tt-xla` (tt-forge-remediation submodule pointer → d82f88636)

## Submodule hashes
| Submodule       | Commit                                   |
|-----------------|------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d82f886362fdd7db8d38da59b45b3046dbe6fdd1 |
| tt-forge-models | eb93515304c0899d7e69514de782e6c8aaaf0d48 |
