# Remediation Summary: hidream-pytorch-Full-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hidream/pytorch-Full-single_device-inference]

## Result
FAIL — pjrt-cpu-fallback-xla-step-marker-device-empty; Tier B compiler-stack bug in partition_fx_graph_for_cpu_fallback path

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-cpu-fallback-xla-step-marker-device-empty

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   IndexError: vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
```
Full traceback:
```
python_package/tt_torch/backend/backend.py:215: in _call_experimental_compile
    self.compiled_graph = bridge.extract_compiled_graph(
torch_xla/_dynamo/dynamo_bridge.py:737: in extract_compiled_graph
    return extract_compiled_graph_helper(xla_model, xla_args)
torch_xla/_dynamo/dynamo_bridge.py:859: in extract_compiled_graph_helper
    return partition_fx_graph_for_cpu_fallback(xla_model, xla_args,
torch_xla/_dynamo/dynamo_bridge.py:804: in partition_fx_graph_for_cpu_fallback
    extract_internal(fused_module), node.args, None)
torch_xla/_dynamo/dynamo_bridge.py:539: in extract_internal
    xla_args_need_update) = extract_graph_helper(xla_model,
torch_xla/_dynamo/dynamo_bridge.py:346: in extract_graph_helper
    torch_xla.sync(reset_scope=False)
torch_xla/torch_xla.py:87: in sync
    torch_xla._XLAC._xla_step_marker(
E   IndexError: vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
```

## Root cause
The HiDream-I1-Full transformer has operations that cannot be lowered to the TT
device, causing `torch.compile` to enter the `partition_fx_graph_for_cpu_fallback`
path in `dynamo_bridge.py`. Within that path, for each CPU-fallback subgraph,
`extract_internal` → `extract_graph_helper` calls `torch_xla.sync()`, which
calls `torch_xla._XLAC._xla_step_marker(device_ordinal=0, ...)`. The C++ side
accesses `devices_[0]` but the device vector is empty (size 0), raising
`std::vector::_M_range_check`.

The XLA device context within the `partition_fx_graph_for_cpu_fallback` iteration
does not have the TT PJRT device registered, so the device list is empty when the
sync is attempted. The CPU inference phase (1:07:46 of model forward pass in
float32) completed successfully; the failure is exclusively in the TT device
compilation stage.

The loader had a pre-existing bug (load_inputs returning raw text strings instead
of tensor inputs for the transformer's forward() method), which has been fixed as
part of this remediation. The residual failure is a separate Tier B compiler-stack
issue.

## Fix
**Loader fix (committed)**:
`hidream/pytorch/loader.py` — `load_inputs` was returning a list of raw text
strings (`["A photo of an astronaut..."]`), but `load_model` returns
`pipeline.transformer` which expects latent/embedding tensors in its `forward()`
call. Fixed `load_inputs` to return a dict of synthetic tensors matching the
`HiDreamImageTransformer2DModel.forward()` signature:
- `hidden_states`: `(batch, in_channels=16, latent_h=16, latent_w=16)`
- `timesteps`: `(batch,)`
- `encoder_hidden_states_t5`: `(batch, 128, t5_channels=4096)`
- `encoder_hidden_states_llama3`: `(num_llama_layers=48, batch, 128, llama3_channels=4096)`
- `pooled_embeds`: `(batch, text_emb_dim=2048)`

**Proposed fix for Tier B bug**:
The `partition_fx_graph_for_cpu_fallback` path in `dynamo_bridge.py` calls
`extract_graph_helper` → `torch_xla.sync()` → `_xla_step_marker` without
ensuring the PJRT TT device is registered in the current XLA runtime context.
The fix would be to ensure the device is properly initialized before the
CPU-fallback partition loop, or to guard the `_xla_step_marker` call when
`_xla_get_default_device()` returns an ordinal that is out of range for the
current device list (e.g., return early instead of asserting).

- Repo: `tt-xla` (or the pinned `torch_xla` package)
- File: `venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py`
  (or the upstream `torch_xla` source that generates this)

## Tier B justification
The `partition_fx_graph_for_cpu_fallback` path in `torch_xla/_dynamo/dynamo_bridge.py`
has a device-state management bug. This requires investigating the XLA runtime's
device registration flow within the dynamo bridge, likely touching the PJRT C++
layer. Root cause diagnosis requires XLA/PJRT expertise; the mechanism is unclear
from the traceback alone.

**Tier B indicator**: `internal-error-unknown-mechanism` — `_xla_step_marker`
fails with an empty device list; the reason the device list is empty in this
sub-context is not apparent from the traceback. Full diagnosis requires inspection
of the C++ PJRT device list management in torch_xla.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    4066.22s (1:07:46) — CPU inference phase ran to completion;
               failure at TT device compilation stage
- Tier A attempts: N/A

## Files changed
- `hidream/pytorch/loader.py` — `load_inputs` now returns tensor dict instead
  of list of strings

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | 643e2b3a044ace6db366b4a1902ef841a30d9b4a   |
| tt-forge-models | 9f8fb77722c15f7fc2175d15d55d7fb433769d27   |
