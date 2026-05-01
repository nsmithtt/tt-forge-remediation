# Remediation Summary: ltxv_0_9_6_gguf/pytorch-0.9.6_dev_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ltxv_0_9_6_gguf/pytorch-0.9.6_dev_Q4_0-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer Error code: 13 in partition_fx_graph_for_cpu_fallback after two loader/frontend fixes

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure: `torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded`

After loader fix, second failure: `NameError: name 'L' is not defined` while executing `%_guards_fn`.

After tt-xla fix, terminal failure: `ValueError: Error code: 13` in `partition_fx_graph_for_cpu_fallback`.

## Root cause

**Bug 1 (loader) — fixed:** `LTXVideoTransformer3DModel.from_single_file` with `GGUFQuantizationConfig` loads the model with `GGUFParameter` tensor subclasses. `GGUFParameter.__torch_function__` calls `super().__torch_function__()` which under TorchDynamo tracing recurses infinitely → `RecursionError`. Fix: call `_dequantize_gguf_and_restore_linear`, clear `_hf_quantizer`/`is_quantized`, and cast to dtype so the model is a plain `nn.Module` before compilation.

**Bug 2 (tt-xla) — fixed:** `ep.module()` with `check_guards=True` (the default) inserts a `_guards_fn` submodule whose generated code references `L` (Dynamo's locals dict). When guard expressions reference keys that are not model inputs, the `L` substitution is incomplete. This causes `NameError: name 'L' is not defined` during AOT re-export inside `run_decompositions` and again when the decomposed module is unlocked. Fix: pass `check_guards=False` to both `ep.module()` call sites in `torch_pass_pipeline`.

**Bug 3 (Tier B) — unfixed:** After the above fixes, `diffusers/models/transformers/transformer_ltx.py:93-95` uses `unflatten(2, (attn.heads, -1))` which causes a graph break. During CPU fallback partitioning, `extract_compiled_graph` → `partition_fx_graph_for_cpu_fallback` → `_xla_warm_up_cache` raises `ValueError: Error code: 13`. This is the known `pjrt-device-to-host-transfer` Tier B bug — transferring TT tensors to the host during the fallback partitioner is unsupported infrastructure.

## Fix

**Loader fix** (`tt_forge_models/ltxv_0_9_6_gguf/pytorch/loader.py`):
- Added import of `_dequantize_gguf_and_restore_linear` from `diffusers.quantizers.gguf.utils`
- In `load_model`, after `from_single_file`: call `_dequantize_gguf_and_restore_linear(self._transformer)`, clear `self._transformer._hf_quantizer = None` and `self._transformer.is_quantized = False`, then call `self._transformer.to(dtype)`

**tt-xla fix** (`tt-xla/python_package/tt_torch/backend/backend.py`):
- In `torch_pass_pipeline`, renamed the local `program` from `torch.export.export` to `_exported`
- Patched `_exported.module = lambda **kw: _orig_module(check_guards=False, **kw)` before calling `_exported.run_decompositions(decompositions)`
- Changed `compiled_graph = program.module()` to `compiled_graph = program.module(check_guards=False)`

**Proposed fix for terminal bug**: The `unflatten` op in `transformer_ltx.py` causes a graph break that triggers the CPU fallback partitioner. The `_xla_warm_up_cache` call in `extract_graph_helper` attempts to transfer TT tensors for the fallback subgraph. Fix requires new infrastructure in the PJRT transfer layer to handle on-device tensors properly during fallback partitioning.

## Tier B justification
Indicator: **new-infrastructure** — `partition_fx_graph_for_cpu_fallback` requires the PJRT runtime to support transferring TT device tensors during graph partitioning, which is not implemented. The Error code: 13 indicates an unsupported device-to-host transfer path. Fixing this requires changes to the PJRT bridge's graph-partitioning infrastructure.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 58.60s (final failing run)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/ltxv_0_9_6_gguf/pytorch/loader.py` (loader fix, tt_forge_models repo)
- `tt-xla/python_package/tt_torch/backend/backend.py` (tt-xla frontend fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | dd55716e075456e53ac7ceab1228d0efa5fad65d |
| tt-forge-models | ff9b29bbdeb2521dc9dbe8bcd7c1d3b87d4b114b |
