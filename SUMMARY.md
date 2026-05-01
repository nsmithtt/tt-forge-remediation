# Remediation Summary: kokoro_tts-pytorch-82M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kokoro_tts/pytorch-82M-single_device-inference]

## Result
FAIL — terminal Tier B: packed LSTM over XLA lazy tensors after dynamo graph break

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
E   RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
If you're using torch.compile/export/fx, it is likely that we are erroneously tracing into a custom kernel. To fix this, please wrap the custom kernel into an opaque custom op.

## Root cause
Two issues:

**Loader (fixed):** No `requirements.txt` existed in `kokoro_tts/pytorch/`, so
the test runner never installed the `kokoro` package, causing
`ModuleNotFoundError: No module named 'kokoro'` at load time.

**Compiler (Tier B, unfixed):** After installing `kokoro`, the test progresses
to inference and hits a dynamo graph break in `kokoro/model.py:100`
(`forward_with_tokens`) at `text_mask = torch.arange(input_lengths.max())`
— `input_lengths.max()` is data-dependent and cannot be traced. In the
resumed eager execution, `DurationEncoder.forward` (inside
`kokoro/modules.py:167`) calls `pack_padded_sequence` followed by `nn.LSTM`,
which dispatches via `_VF.lstm(input, batch_sizes, ...)`. The `batch_sizes`
parameter causes the packed-sequence path, which tries to access raw tensor
storage. XLA lazy tensors are not yet materialized at that point, producing
`RuntimeError: The tensor has a non-zero number of elements, but its data is
not allocated yet.`

The full call chain: `torch_overrides.py:34 __torch_function__` →
`torch/nn/modules/rnn.py:1139 _VF.lstm` → crash.

## Fix
**Loader fix (committed):** Added `kokoro_tts/pytorch/requirements.txt`
containing `kokoro` to `tt-forge-models` on branch
`remediation/kokoro_tts-pytorch-82M-single_device-inference`.

**Compiler fix (proposed):** The fundamental issue is that after a dynamo graph
break, eager execution of packed LSTM (`_VF.lstm` with `batch_sizes`) on XLA
lazy tensors fails because the lazy evaluator has not yet materialized the
tensors. A fix would require either (a) synchronizing all XLA tensors before
any op that accesses raw storage in resumed-eager mode, or (b) implementing
a proper packed-sequence LSTM lowering in tt-xla that does not rely on
raw C++ tensor storage access. This is new infrastructure work in `tt-xla`.

## Tier B justification
**new-infrastructure**: The fix requires either a general mechanism for
synchronizing lazy tensors before ops that access raw tensor storage in
dynamo graph-break resumed-eager contexts, or a new packed LSTM lowering path
in tt-xla. Neither exists today and both require non-trivial new infrastructure.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    90.47s (1:01:30 for full run including kokoro install)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: kokoro_tts/pytorch/requirements.txt` (added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d05e140599008110e160ea3ad8d119ba56486d3d |
