# Remediation Summary: llamafactory-image_text_to_text-pytorch-tiny_random_Llama_4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llamafactory/image_text_to_text/pytorch-tiny_random_Llama_4-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer (Error code: 13) during partition_fx_graph_for_cpu_fallback after all loader fixes and Tier A slice-clamp fix

## Stack layer
tt-xla

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
```
ValueError: Error code: 13
```
in `torch_xla._XLAC._xla_warm_up_cache` called from `partition_fx_graph_for_cpu_fallback` →
`extract_internal(fused_module)` → `extract_graph_helper` → `_xla_warm_up_cache`.

Original failure was:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
NotImplementedError: Cannot copy out of meta tensor; no data!
```

## Root cause
Four distinct loader bugs were fixed sequentially:

1. **Meta-tensor freqs_ci** — `Llama4VisionRotaryEmbedding.__init__` sets `self.freqs_ci`
   as a plain Python attribute (not `register_buffer`). Under transformers 5.x
   `init_empty_weights` (default meta-device init), it becomes a meta tensor and
   calling `.to(device)` in `forward` raises `NotImplementedError`.

2. **Complex RoPE** — TT device does not support `complex64`. Both text and vision
   RoPE in Llama4 use `torch.polar` / `view_as_complex`. Patches decompose to
   real `(cos, sin)` arithmetic.

3. **Boolean-mask boolean-index in get_placeholder_mask** — `torch_compilable_check`
   inside the original `get_placeholder_mask` performs `inputs_embeds[special_image_mask]`
   (data-dependent boolean indexing) during XLA graph capture → Error code: 13 in
   `_xla_warm_up_cache`.

4. **aten.im2col (Llama4UnfoldConvolution)** — `torch.nn.Unfold` lowers to
   `aten.im2col` → `stablehlo.reduce_window` with a non-Add/Max body that
   tt-mlir cannot legalize. Since `stride == kernel_size`, a semantically
   equivalent `reshape + permute` avoids the op.

After these four loader fixes, a Tier A tt-xla compiler fix was applied:

5. **aten.slice OOB start (Tier A)** — `StaticSlidingWindowLayer.update` slices with
   `full_value_states[:, :, -sliding_window+1:, :]` where `sliding_window=8192`
   but the tensor has only 2339 elements in that dimension. XLA raises
   `"Value out of range (expected to be in range of [-2339, 2338], but got -8191)"`.
   PyTorch eager silently clamps OOB negative starts to 0; XLA validates strictly.
   Fixed in `TorchFunctionOverride.__torch_function__` by pre-clamping start/end to
   `[-dim_size, dim_size]` before XLA sees the index.

After the slice fix, the terminal failure is **pjrt-device-to-host-transfer** (Tier B):
`ValueError: Error code: 13` in `_xla_warm_up_cache` during
`partition_fx_graph_for_cpu_fallback`. The most likely cause is `aten.topk` and/or
`aten.scatter_.src` in `Llama4Router.forward`:

```python
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value)
```

If `aten.topk` is not supported on TT device, `partition_fx_graph_for_cpu_fallback`
assigns it to a CPU subgraph. That subgraph's input (`router_logits`) is a TT tensor,
so transferring it to CPU requires a device-to-host operation → Error code: 13 in
the XLA warm-up cache trace.

## Fix
Fixes already applied and pushed:

**tt-forge-models** (`remediation/llamafactory-image_text_to_text-pytorch-tiny_random_Llama_4-single_device-inference`):
- `llamafactory/image_text_to_text/pytorch/loader.py`:
  - Re-initialize `freqs_ci` for meta-tensor `Llama4VisionRotaryEmbedding` modules
    by calling `Llama4VisionRotaryEmbedding.__init__(module, vision_config)` and
    decomposing result into real `(cos_freqs, sin_freqs)` components
  - Patch `Llama4TextRotaryEmbedding.forward` to return `_RoPETuple([cos, sin])`
  - Patch `apply_rotary_emb` and `vision_apply_rotary_emb` to accept tuple
    `(cos, sin)` and compute RoPE with real arithmetic
  - Patch `Llama4ForConditionalGeneration.get_placeholder_mask` to skip the
    `torch_compilable_check` boolean-index validation
  - Patch `Llama4UnfoldConvolution.forward` with a `reshape + permute` decomposition

**tt-xla** (`remediation/llamafactory-image_text_to_text-pytorch-tiny_random_Llama_4-single_device-inference`):
- `python_package/tt_torch/torch_overrides.py`:
  - In `TorchFunctionOverride.__torch_function__`, pre-clamp `aten.slice.Tensor`
    `start`/`end` arguments to `[-dim_size, dim_size]` when the dimension size is
    statically known, matching PyTorch eager semantics

Proposed fix for the terminal Tier B bug:
Support `aten.topk` and `aten.scatter_.src` on TT device in the PJRT/tt-mlir
compilation path so these ops do not fall through to `partition_fx_graph_for_cpu_fallback`.
Alternatively, improve the CPU-fallback partitioner to handle device-to-host tensor
transfers for mixed CPU/TT subgraphs without triggering Error code: 13.

## Tier B justification
- **Indicator**: new-infrastructure
- `partition_fx_graph_for_cpu_fallback` + `_xla_warm_up_cache` fails with Error
  code: 13 (device-to-host transfer unsupported) when a CPU-fallback subgraph
  receives TT tensors as input. Fixing requires either (a) implementing `aten.topk`
  natively on TT device to avoid the CPU fallback entirely, or (b) adding proper
  device-to-host transfer infrastructure in the PJRT layer.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 62.63s (after Tier A fix applied)
- Tier A attempts: 1

## Files changed
**tt-forge-models** (4 loader commits):
- `llamafactory/image_text_to_text/pytorch/loader.py`

**tt-xla** (2 commits):
- `python_package/tt_torch/torch_overrides.py` (Tier A slice-clamp fix)
- `third_party/tt_forge_models` (submodule pointer update)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d95fb1a9b3692bdcd892f2bf50bdeab7b48df1e4 |
| tt-forge-models | 130b3e1cc7b3e64c515511287a1981ba6cd5000d |
