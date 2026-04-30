# Remediation Summary: kokoro_tts-pytorch-82M-v1.1-zh-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kokoro_tts/pytorch-82M-v1.1-zh-single_device-inference]

## Result
FAIL — packed LSTM (_VF.lstm with batch_sizes) is incompatible with XLA lazy tensors after a graph break; requires new infrastructure

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
lstm-packed-sequence-dynamo-tracing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
If you're using torch.compile/export/fx, it is likely that we are erroneously tracing into a custom kernel. To fix this, please wrap the custom kernel into an opaque custom op.
```

## Root cause

Two loader bugs (missing requirements) gate a compiler bug:

**Loader bugs** (fixed):
1. `kokoro` package was not in `requirements.txt` → `ModuleNotFoundError: No module named 'kokoro'`
2. `misaki[zh]` extra was not in `requirements.txt` → cascading `ModuleNotFoundError` for `ordered_set` / `pypinyin` when the Chinese (`lang_code="z"`) pipeline is initialized

After fixing the requirements, the test advances to a compiler-stack failure:

**Compiler bug (Tier B)**:
`kokoro/model.py:100` causes a graph break in `forward_with_tokens`:
```python
text_mask = torch.arange(input_lengths.max()).unsqueeze(0)...
```
`input_lengths.max()` is data-dependent, so torch.compile/dynamo breaks the graph here and resumes execution eagerly.  After the break, `self.predictor.text_encoder(...)` calls `DurationEncoder.forward` (`kokoro/modules.py:148`), which runs
```python
x = nn.utils.rnn.pack_padded_sequence(x, lengths, ...)
x, _ = block(x)   # block is nn.LSTM
```
with `x.data` still on the XLA lazy device.  The `_VF.lstm(input, batch_sizes, ...)` packed-sequence path tries to access the raw tensor memory directly in C++; because XLA tensors are lazy (data not yet materialized), this raises "The tensor has a non-zero number of elements, but its data is not allocated yet."

`TextEncoder.forward` (`kokoro/modules.py:50`) has an identical pattern and would trigger the same error if reached.

## Fix

**Loader fix committed** (`tt_forge_models` remediation branch):
- Created `kokoro_tts/pytorch/requirements.txt` containing `kokoro` and `misaki[zh]`.

**Compiler fix (not attempted — Tier B)**:
The XLA backend needs either:
(a) A lowering for the packed form of `aten.lstm` / `_VF.lstm(input, batch_sizes, ...)` so the op can be dispatched to XLA without materializing lazy tensors, OR
(b) Support for dynamic-shape tracing that eliminates the graph break at `input_lengths.max()`.
Both require new infrastructure in tt-xla / torch_xla.

## Tier B justification

**new-infrastructure**: Supporting packed-sequence LSTM in the XLA lazy-tensor model requires adding a new op lowering (aten.lstm packed path) that can be composed from XLA primitives, or extending the dynamic-shape machinery to trace through data-dependent `torch.arange` calls without a graph break. Neither is a scoped single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    105.96s (1:45) to reach the compiler bug after fixing requirements
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/kokoro_tts/pytorch/requirements.txt` — created (kokoro + misaki[zh])

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 43ca6ac735e4f663b725f9d12ffd6ef5aacd7535 |
| tt-forge-models | 2307b2b08d24c25c326886641b2110b36c9c2df6 |
