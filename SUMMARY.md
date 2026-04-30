# Remediation Summary: kronos-pytorch-small-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kronos/pytorch-small-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
rope-cos-sin-non-buffer-device-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_method mul(*(FakeTensor(..., device='xla:0', size=(1, 8, 64, 64), dtype=torch.bfloat16), FakeTensor(..., size=(1, 1, 64, 64), dtype=torch.bfloat16)), **{}): got RuntimeError('Unhandled FakeTensor Device Propagation for aten.mul.Tensor, found two different devices xla:0, cpu')

## Root cause
`RotaryPositionalEmbedding` in `kronos/pytorch/src/module.py` caches `cos_cached` and
`sin_cached` as plain Python attributes (not registered buffers) via `_update_cos_sin_cache`.
When Dynamo traces the model with real XLA tensors, the real XLA tensor is stored in
`self.cos_cached`. During the subsequent FakeTensor abstract-interpretation pass, Dynamo
accesses `self.cos_cached` (a real XLA tensor) as a graph-input leaf and incorrectly
maps it to a CPU FakeTensor — either because `self.seq_len_cached == seq_len` causes the
cache to be reused without triggering re-creation in the FakeTensor context, or because
the PJRT device is not preserved when Dynamo "fakifies" the cached leaf tensor.
The result is a two-device conflict: `q` on `xla:0` multiplied by `cos` on `cpu`.

## Fix
Rewrote `RotaryPositionalEmbedding.__init__` to precompute cos/sin for `max_seq_len=4096`
and register them as non-persistent buffers (`register_buffer(..., persistent=False)`).
Buffers are correctly moved to the target device with the model and are properly
represented in FakeTensor mode. Removed the `_update_cos_sin_cache` method and the
dynamic `seq_len_cached`/`cos_cached`/`sin_cached` attributes entirely. The `forward`
method now slices `self.cos_cached[:, :, :seq_len, :]` to the actual sequence length.

File changed: `tt_forge_models/kronos/pytorch/src/module.py`
Repo: `tenstorrent/tt-forge-models` on branch
`remediation/kronos-pytorch-small-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    50.33s
- Tier A attempts: N/A

## Files changed
- tt_forge_models/kronos/pytorch/src/module.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 916989799117a6659f161954ffc2b862f627ed8e |
| tt-forge-models | 6d61d103f9bbb6a404d7b42f7a54679e822c49cd |
