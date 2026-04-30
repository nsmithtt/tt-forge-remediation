# Remediation Summary: h2ovl_mississippi-pytorch-2B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[h2ovl_mississippi/pytorch-2B-single_device-inference]

## Result
FAIL — model loads and runs on TT silicon (pcc=0.9848) but falls below required pcc=0.99; root cause of the precision gap is unknown and requires compiler-stack investigation

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
ttmlir-vlm-bf16-pcc-gap

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
RuntimeError: Tensor.item() cannot be called on meta tensors
```
After fix 1, second failure:
```
AttributeError: 'H2OVLChatModel' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?
```
After fix 2, remaining failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9848028117098736. Required: pcc=0.99.
```

## Root cause
Three bugs were found, two in the loader and one unresolved in the compiler stack.

**Bug 1 (loader):** transformers 5.x uses `torch.device("meta")` context when initialising models in `from_pretrained`. The remote `InternVisionEncoder.__init__` calls `torch.linspace(...).item()` to build drop-path rate lists, which fails on meta tensors. The existing remediation commit `a8027d12ca` patches `torch.Tensor.item` to return `0.0` for meta scalars during the `from_pretrained` call. The 0.0 values are irrelevant at inference time because `DropPath` is a no-op when `model.eval()` is set.

**Bug 2 (loader):** transformers 5.x expects every `PreTrainedModel.__init__` to call `self.post_init()` at the end, which sets `self.all_tied_weights_keys`. The remote `H2OVLChatModel.__init__` predates this requirement and omits the call. When `_finalize_model_loading` later accesses `model.all_tied_weights_keys`, it raises `AttributeError`. Patching `H2OVLChatModel.__init__` directly via `importlib.import_module` fails because `AutoModel.from_pretrained` resolves the class via `get_class_in_module`, which uses a different `sys.modules` key and creates a distinct class object. The fix patches `PreTrainedModel._finalize_model_loading` (a staticmethod) to call `model.post_init()` when `all_tied_weights_keys` is absent, then restores the original in a `finally` block.

**Bug 3 (compiler stack, unresolved):** After the two loader fixes, the model loads and runs on TT silicon and produces output. However, the PCC between TT bfloat16 and CPU bfloat16 is 0.9848, below the required 0.99. Both runs use the same bfloat16 dtype (the test framework applies `dtype_override=torch.bfloat16` to both model and inputs before CPU reference and TT compilation). The ~1.5% gap is unexplained; the root mechanism (which op or layer accumulates precision error) was not identified.

## Fix
**Fix 1** — already in `tt_forge_models` commit `a8027d12ca`:
- `h2ovl_mississippi/pytorch/loader.py`: patch `torch.Tensor.item` to return `0.0` for meta tensors during `from_pretrained`.

**Fix 2** — new commit `e0d5fc83c6` on `remediation/h2ovl_mississippi-pytorch-2B-single_device-inference`:
- `h2ovl_mississippi/pytorch/loader.py`: import `PreTrainedModel`; patch `PreTrainedModel._finalize_model_loading` to call `model.post_init()` when `all_tied_weights_keys` is absent; restore original in `finally`.

**Fix 3 (proposed):** Profile layer-by-layer outputs between TT and CPU to identify which op or attention/LayerNorm path causes the 1.5% PCC gap. The fix would live in tt-mlir or tt-metal depending on the identified op.

## Tier B justification
Which indicator: `internal-error-unknown-mechanism`

The precision gap (pcc=0.9848 vs required 0.99) is a real discrepancy between TT bfloat16 and CPU bfloat16 execution on this 2B vision-language model. The exact layer or operation responsible was not identified. Diagnosing the root cause requires layer-by-layer output profiling across a complex multi-modal model (vision encoder + projection MLP + Mistral decoder), which is beyond the scope of a single scoped fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    248.84s (0:04:08)
- Tier A attempts: N/A

## Files changed
- `h2ovl_mississippi/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 94362e631 |
| tt-forge-models | e0d5fc83c6 |
