# Remediation Summary: gemma3_4b_it_logiqa_dpo_c_new-causal_lm-pytorch-Gemma3_4B_IT_LogiQA_DPO_C_new-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_4b_it_logiqa_dpo_c_new/pytorch-Gemma3_4B_IT_LogiQA_DPO_C_new-single_device-inference]

## Result
FAIL — fix applied (tt-xla __getitem__ slice clamp), silicon verification blocked by gated base model

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-32, 31], but got -1023)

## Root cause
The model uses `google/gemma-3-4b-it` (sliding_window=1024) loaded with `max_length=32`
(seq_len=32). transformers 5.2.0 `DynamicSlidingWindowLayer.update()` does:

```python
self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
```

With sliding_window=1024 and seq_len=32, `start = -1023`. PyTorch eager silently
clamps this; the XLA lazy backend raises "Value out of range" because `-1023 < -32`.

`TorchFunctionOverride.__torch_function__` in tt-xla already intercepted
`torch.ops.aten.slice.Tensor` to clamp out-of-bounds negative indices
(commit `ee94c31a4`). However, when `tensor[:, :, -1023:, :]` is evaluated on
an XLA tensor in eager mode, `TorchFunctionMode` receives `func =
torch.Tensor.__getitem__`, NOT `torch.ops.aten.slice.Tensor`. The existing
`aten.slice.Tensor` branch was therefore never entered for this model, and the
error propagated into the XLA C++ dispatch layer.

The same `aten.slice.Tensor` fix had worked for the Albert Wesker (albert-wesker-gguf)
model because an older transformers cache implementation explicitly called
`aten.slice.Tensor`/`narrow` rather than `__getitem__`.

## Fix
Added a `torch.Tensor.__getitem__` branch to `TorchFunctionOverride.__torch_function__`
in `tt-xla/python_package/tt_torch/torch_overrides.py`. When all index elements
are plain `slice` objects (no integer indices, no `...`, no boolean masks), the
fix iterates over the slice tuple, identifies any `start` or `stop` that is
a Python `int` less than `-size`, clamps it to `-size`, and reconstructs `args`
before calling through to `func`. This mirrors the existing `aten.slice.Tensor`
clamp logic for the `__getitem__` eager dispatch path.

Standalone verification on XLA device (Blackhole p150b) confirmed `tensor[:, :, -1023:, :]`
on a `[1, 8, 32, 64]` XLA tensor returns `[1, 8, 32, 64]` (all elements kept,
as expected when seq_len < sliding_window). Normal in-range slices (e.g. `[:, :, -32:, :]`)
are unaffected.

Full `pytest` could not be run: base model `google/gemma-3-4b-it` is HuggingFace-gated
and the CI token (`nsmithtt`) does not have authorization.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL (gated model — could not be run)
- Hardware:    blackhole-p150b (standalone XLA slice test only)
- Duration:    not-run
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `torch.Tensor.__getitem__` branch to clamp out-of-bounds negative slice indices in eager XLA mode

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9b2a881cf4487a35c5885944ff1bd6cceab3c1f6 |
| tt-forge-models | 2581f49c6d5a7ac24d44b45f023f5929ab161bac |
