# Remediation Summary: gemma_2_2b_it_q4f16_0_mlc-causal_lm-pytorch-gemma_2_2b_it_q4f16_0_mlc-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_2b_it_q4f16_0_mlc/causal_lm/pytorch-gemma_2_2b_it_q4f16_0_mlc-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-negative-start-xla-rejection

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Value out of range (expected to be in range of [-16, 15], but got -4095)

Triggered during `partition_fx_graph_for_cpu_fallback` in the torch-xla dynamo bridge while
executing `aten.slice.Tensor(%cat_4, 2, -4095, 9223372036854775807)` in XLA eager mode.

## Root cause

Two independent bugs, both needed to be fixed to reach a passing run:

**Bug 1 (loader):** The original loader called
`AutoModelForCausalLM.from_pretrained("mlc-ai/gemma-2-2b-it-q4f16_0-MLC")`, which fails with
`ValueError: Unrecognized model in mlc-ai/gemma-2-2b-it-q4f16_0-MLC. Should have a 'model_type'
key in its config.json`. The MLC HuggingFace repo stores weights in binary shards
(`params_shard_*.bin`) indexed by `ndarray-cache.json` and has no standard `config.json`.
`from_pretrained` cannot handle this format.

**Bug 2 (tt-xla):** Gemma 2 uses sliding-window attention. `StaticSlidingWindowLayer.update()` in
transformers computes `full_value_states[:, :, -sliding_window+1 :, :]` = `[:, :, -4095:, :]` with
`sliding_window=4096`. This produces an `aten.slice.Tensor` node with a negative start index
(-4095) in the FX graph. During `partition_fx_graph_for_cpu_fallback` in the torch-xla dynamo
bridge, the graph is executed with XLA eager tensors. XLA's eager `aten.slice.Tensor` implementation
rejects negative start indices (requires values in [-16, 15]), causing the observed
`RuntimeError: Value out of range`. After `torch.export`, all tensor shapes are concrete, so the
negative start can be statically resolved to the equivalent non-negative index
`max(0, dim_size + start)`.

## Fix

**Fix 1 — loader (`tt_forge_models`):**
Rewrote `gemma_2_2b_it_q4f16_0_mlc/causal_lm/pytorch/loader.py` to implement a complete MLC
q4f16_0 binary weight parser:
- `_make_gemma2_config()`: Constructs `Gemma2Config` matching the 2B IT architecture directly
  (26 layers, 2304 hidden, sliding_window=4096, attn_logit_softcapping=50.0, etc.)
- `_get_weight_index()`: Parses `ndarray-cache.json` to build a `{name: record}` lookup table
- `_read_tensor()`: Downloads and reads raw binary data from shard files using byte offsets
- `_unpack_int4()`: Unpacks 8 signed 4-bit values per uint32 (little-endian nibble order)
- `_dequantize_linear()` / `_dequantize_embed()`: Full dequantization using int4 weights and
  float16 group-size-32 scales
- `_load_mlc_q4f16_gemma2()`: Constructs the HF Gemma2 model from config, dequantizes all weights
  (splitting `qkv_proj` → `q_proj/k_proj/v_proj` and `gate_up_proj` → `gate_proj/up_proj`),
  loads via `load_state_dict(strict=False)`, and calls `tie_weights()`

Branch: `remediation/gemma_2_2b_it_q4f16_0_mlc-causal_lm-pytorch-gemma_2_2b_it_q4f16_0_mlc-single_device-inference`
in `third_party/tt_forge_models` (tt-xla submodule), commit `e974f27c8f`.

**Fix 2 — tt-xla FX pass (Tier A):**
Added `clamp_out_of_range_slice_starts()` FX pass to
`python_package/tt_torch/backend/passes.py`. The pass iterates all `aten.slice.Tensor` nodes in
the FX graph, detects negative integer constant start indices, reads the concrete dimension size
from `node.meta['val'].shape[dim]`, and replaces the start with `max(0, dim_size + start)`.
This is safe because `torch.export` produces static shapes for all tensors.

Wired the new pass into `torch_pass_pipeline()` in
`python_package/tt_torch/backend/backend.py` (called after `bypass_assert_tensor_metadata`,
before `compiled_graph.recompile()`).

Branch: `remediation/gemma_2_2b_it_q4f16_0_mlc-causal_lm-pytorch-gemma_2_2b_it_q4f16_0_mlc-single_device-inference`
in `tt-xla`, commit `7428d4e5d0`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    231.43s (0:03:51)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma_2_2b_it_q4f16_0_mlc/causal_lm/pytorch/loader.py` (rewritten)
- `tt-xla/python_package/tt_torch/backend/passes.py` (added `clamp_out_of_range_slice_starts`)
- `tt-xla/python_package/tt_torch/backend/backend.py` (import + call of new pass)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7428d4e5d06f7d3c108539280740d26089c74550 |
| tt-forge-models | e974f27c8fddb3f2e6b4842de46b06a59e4e8f3e |
