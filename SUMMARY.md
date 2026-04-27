# AlinVLM v1.3 HF Bringup — Fix Summary

**Test:** `tests/runner/test_models.py::test_all_models_torch[alinvlm/pytorch-v1_3-single_device-inference]`
**Result:** SILICON_PASS

## Root causes and fixes

AlinVLM uses Qwen3VLForConditionalGeneration (a multimodal transformer). Four separate issues blocked inference on TT silicon.

### 1. SpeculationLogDivergence (`_stabilize_forge_models_sys_modules`)

`DynamicLoader.import_model_loader` registers loaders under `tt-forge-models.*` keys in `sys.modules`.  Dynamo's `LOAD_GLOBAL` handler calls `import_source(fn.__module__)` for every function it encounters during tracing.  `fn.__module__` uses underscores (`tt_forge_models.*`), which was absent from `sys.modules`, causing Python to re-import the loader from disk mid-trace.  That re-ran `nn.Module.__getattr__ = patched_getattr` (from the MiniCPM-V loader), creating a new function object; Dynamo's speculation log had recorded the old one and raised `SpeculationLogDivergence`.

**Fix:** Call `_stabilize_forge_models_sys_modules()` in `load_model()` before the first `torch.compile` trace.  It mirrors every `tt-forge-models.*` entry to `tt_forge_models.*` and adds synthetic namespace stubs so `import_source` becomes a no-op.

### 2. TT L1 overflow and unsupported Conv3d (`_patched_visual_forward`)

`Qwen3VLVisionModel` uses `nn.Conv3d` for patch embedding and processes sequences up to 11,008 tokens, both of which exceed TT hardware limits (L1 max ~1.5 MB per core; no Conv3d support).

**Fix:** Decorate `Qwen3VLVisionModel.forward` with `@torch.compiler.disable(recursive=True)` and move the visual encoder to CPU before calling the original forward.  Also cap `max_pixels` at 28x28 (the minimum valid grid for 14x14 patches + 2x2 spatial merge): 4 patches -> 1 merged token -> ~17 total tokens -> ~140 KB of L1.

### 3. LRU cache overflow from `.tolist()` graph breaks (`_get_image_features_eager`)

`Qwen3VLModel.get_image_features` calls `image_grid_thw.prod(-1).tolist()` to compute split sizes.  Each `aten._local_scalar_dense` failure added a new graph break, causing 15+ recompilations that filled and overflowed the XLA LRU computation cache.

**Fix:** Wrap `get_image_features` so the inner call runs through a `@torch.compiler.disable` helper, preventing the `.tolist()` from entering the compiled graph entirely.

### 4. LRU cache overflow from boolean-index scatter in decoder loop (`_patched_deepstack_process`)

`Qwen3VLTextModel._deepstack_process` runs inside a per-decoder-layer loop and contains:
```python
local_this = hidden_states[visual_pos_masks, :] + visual_embeds
hidden_states[visual_pos_masks, :] = local_this  # index_put
```
Boolean indexing with `visual_pos_masks` produces tensors whose shape equals `visual_pos_masks.sum()` — a data-dependent value.  Each unique value compiled a new XLA computation.  Called repeatedly per layer, this flooded the LRU cache and evicted earlier entries, causing the observed `RuntimeError: Check failed: cachedComputation`.

**Fix:** Patch `Qwen3VLTextModel._deepstack_process` with `@torch.compiler.disable(recursive=True)` so it runs eagerly, never entering the XLA compilation path.

### 5. RoPE index computation on TT device (`_patched_get_rope_index`)

`Qwen3VLModel.get_rope_index` uses `input_ids.tolist()` and Python list control flow that cannot run on TT device tensors.

**Fix:** Decorate with `@torch.compiler.disable`, move all inputs to CPU before calling the original method, and move the returned `position_ids` back to the original device so the subsequent RoPE matmul sees consistent devices.

## Files changed

- `tt-xla/third_party/tt_forge_models/alinvlm/pytorch/loader.py` — all five patches above
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — changed `alinvlm/pytorch-v1_3` from `KNOWN_FAILURE_XFAIL` to `EXPECTED_PASSING`

## Branches

- **tt-forge-models**: `remediation/anima-fp8-nvfp4mixed-fix`
- **tt-xla**: `report/alinvlm-pytorch-v1_3-hf-bringup`
