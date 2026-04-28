# Remediation Summary: dinerburger_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_IQ4_NL_GGUF-single_device-inference

## Skill version
14

## Test
tests/runner/test_models.py::test_all_models_torch[dinerburger_qwen3_5_35b_a3b_gguf/causal_lm/pytorch-35B_A3B_IQ4_NL_GGUF-single_device-inference]

## Result
FAIL — segfault in TorchFunctionOverride.__torch_function__ during partition_fx_graph_for_cpu_fallback while probing in-place ops inside torch_chunk_gated_delta_rule

## Failure
Original reported failure: Fatal Python error: Segmentation fault

After loader fix, the segfault moved to the compilation stage:
```
Fatal Python error: Segmentation fault

Current thread 0x... (most recent call first):
  File ".../tt_torch/torch_overrides.py", line 34 in __torch_function__
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 859 in extract_compiled_graph_helper
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 737 in extract_compiled_graph
  File ".../tt_torch/backend/backend.py", line 215 in _call_experimental_compile
  ...
  File ".../transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py", line 1487 in forward
```

## Root cause
Two independent bugs:

**Bug 1 (loader layer — fixed):** Two loader-layer failures prevented the model from loading:

1. `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — 26 GGUF loaders
   that patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` use the
   old 2-argument signature. Transformers 5.2.0 added a `model_to_load=` keyword argument
   and calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`. Since all loaders are
   imported during test collection, whichever loader runs last installs its stale patched
   function globally, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected
   keyword argument 'model_to_load'` for all tests that hit `from_pretrained` with a GGUF file.

2. The dinerburger loader was missing `_patch_transformers_qwen35moe_gguf()`. The GGUF file
   declares architecture `qwen35moe` which transformers 5.x does not natively support for
   GGUF loading. Without the patch, `load_gguf_checkpoint` raises `ValueError: GGUF model
   with architecture qwen35moe is not supported yet` and the expert tensor key mapping
   (`blk.N.ffn_gate_exps` / `blk.N.ffn_up_exps`) is missing.

**Bug 2 (compiler-stack layer — NOT fixed):** After the loader is repaired, the model
compiles into the tt-xla torch.compile pipeline. `Qwen3_5_MoeForCausalLM.forward` contains
linear-attention layers that call `torch_chunk_gated_delta_rule`, a pure-Python function
with two nested Python `for` loops and in-place tensor mutations (`masked_fill_`, in-place
slice assignments `attn[..., i, :i] = ...`). When the XLA dynamo bridge runs
`partition_fx_graph_for_cpu_fallback` to probe which ops can execute on TT silicon, it
calls `TorchFunctionOverride.__torch_function__` (tt_torch/torch_overrides.py:34) on each
node in the partitioned FX graph. One of these calls triggers a segfault — likely due to a
tensor aliasing or mutation invariant violated by the in-place op that was previously
recorded as a graph node, then replayed during the partitioning probe.

The crash is in the **tt-xla compiler frontend** (`tt_torch/torch_overrides.py` /
`torch_xla/_dynamo/dynamo_bridge.py`).

## Fix
**Bug 1 (loader layer — applied):**

- Updated all 26 GGUF loaders whose `_patched_load_gguf_checkpoint` used the old
  `(gguf_path, return_tensors=False)` signature to `(*args, **kwargs)`, forwarding all
  arguments to `_orig_load_gguf_checkpoint`. This is not a forbidden workaround — it is
  a signature compatibility fix for a transformers 5.2.0 API change.

- Added `_patch_transformers_qwen35moe_gguf()` to the dinerburger loader (mirrors the
  amarck_qwen3_5_35b_a3b_abliterated_gguf loader which uses the same model family):
  - Registers `qwen35moe` in `GGUF_SUPPORTED_ARCHITECTURES` and
    `GGUF_TO_TRANSFORMERS_MAPPING["config"]`
  - Adds `Qwen35MoeTensorProcessor` (subclass of `Qwen2MoeTensorProcessor`) that uses
    `.get()` for missing keys instead of raising `KeyError` for out-of-range layers
  - Patches `load_gguf_checkpoint` to remap `model_type qwen35moe → qwen3_5_moe_text`
    and synthesise `layer_types` from `full_attention_interval`
  - Patches `get_gguf_hf_weights_map` to add `ffn_gate_exps` / `ffn_up_exps` keys for
    the split expert tensors (GGUF stores them separately; gguf-py only maps the fused
    `gate_up_proj` key)

**Bug 2 (compiler-stack bug — proposed fix):**

The `partition_fx_graph_for_cpu_fallback` in `torch_xla/_dynamo/dynamo_bridge.py` probes
nodes by replaying them through `TorchFunctionOverride.__torch_function__`. When an FX
graph node recorded an in-place mutation (`masked_fill_`, `__setitem__`), the probe
replays the op on the already-mutated tensor, potentially violating aliasing invariants
that the underlying C extension assumes. The fix should be in **tt-xla** (or the upstream
`torch_xla` dynamo bridge):

1. **Guard in `torch_overrides.py`**: before calling `func(*args, **kwargs)` at line 34,
   check whether the call is from inside `partition_fx_graph_for_cpu_fallback` (or more
   generally, during graph partitioning probe), and skip in-place variants.

2. **Clone tensors before probing in `dynamo_bridge.py`**: in
   `partition_fx_graph_for_cpu_fallback`, create fresh copies of any mutated tensor
   arguments before each probe call so the replay starts from a clean state.

3. **Disable probing for known in-place-heavy custom ops**: detect Python functions that
   use in-place ops in loops (e.g. via `torch._dynamo.disable` on the partitioner side,
   not on the model side) and treat them as opaque CPU fallback without probing.

## Verification
pytest exited with segfault (Fatal Python error) after the loader fix. Test did not reach
PASS. Hardware: n150/n300 (arch-c-36 config).

## Files changed
- `tt-xla/third_party/tt_forge_models/dinerburger_qwen3_5_35b_a3b_gguf/causal_lm/pytorch/loader.py`
  — replaced stub qwen35 patch with full qwen35moe GGUF patch
- 26 other GGUF loaders in `tt-xla/third_party/tt_forge_models/` — updated
  `_patched_load_gguf_checkpoint` signature from `(gguf_path, return_tensors=False)` to
  `(*args, **kwargs)`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 60c9d8b19370821835350ef7227f930583c99a39 |
| tt-forge-models | 567df1af7e2c21732fba6234196d389283e893da |
