# Remediation Summary: internlm_xcomposer2-pytorch-VL_7B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[internlm_xcomposer2/pytorch-VL_7B-single_device-inference]

## Result
FAIL — NaN output from TT device after all loader fixes applied; root cause unknown without further per-op debugging

## Stack layer
tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-nan-output-internlm-xcomposer2-vl

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original:
```
RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations:
prims::view_of(Tensor(a) a) -> Tensor(a). We only support functionalizing operators
whose outputs do not have alias annotations (e.g. 'Tensor(a)' is a Tensor with an alias
annotation whereas 'Tensor' is a Tensor without. The '(a)' is the alias annotation).
```

After fixing prims::view_of and the CLIPVisionTower meta-init issues:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=nan (invalid value). Required: pcc=0.99.
```

## Root cause

**Bug 1 (fixed — loader, tt-xla): prims::view_of alias annotation**

`squeeze` decomposes to `prims.squeeze + prims.view_of`. `prims::view_of` has alias
annotations `Tensor(a) -> Tensor(a)` that `partition_fx_graph_for_cpu_fallback`'s
functionalization layer cannot handle. Fix: `bypass_prims_view_of` FX pass in
`tt-xla/python_package/tt_torch/backend/passes.py` replaces every `prims::view_of`
node with its input (semantically a no-op for inference).

**Bug 2 (fixed — loader): CLIPVisionTower meta device context**

`transformers 5.x` initializes all models inside a `torch.device("meta")` context
(via `init_empty_weights()`). `InternLMXComposer2ForCausalLM.__init__` calls
`build_vision_tower()` → `CLIPVisionTower.__init__` → `load_model()` →
`CLIPVisionModel.from_pretrained()`, which raises
`"You are using from_pretrained with a meta device context manager"`.

The meta context is pushed via `TorchFunctionMode` stack, not via
`torch.set_default_device()`, so `torch.set_default_device(None)` does not
escape it. Fix: patch `CLIPVisionTower.load_model` to build CLIP from config only
(no `from_pretrained`), setting `image_size = 35 * patch_size = 490` so the
structure has 1226 position embeddings matching the VL checkpoint's pre-resized
embeddings. Patch `resize_pos` to a no-op during init; call it manually after
`from_pretrained` completes (it detects the embeddings are already the right size
and skips interpolation).

**Bug 3 (unfixed — Tier B): NaN output on TT device**

After both loader fixes, the model loads and runs inference to completion
(~567s) but produces NaN output. The mechanism is unknown without per-op
debugging. Likely candidates: BF16 overflow in CLIP visual attention over
1226-token sequences, or numerical instability in InternLM2's rotary embeddings
at long context. The root cause spans multiple layers (CLIP visual encoder +
InternLM2 decoder), making it a cross-cutting precision issue.

## Fix

**Bug 1 — tt-xla:**
- Added `bypass_prims_view_of(gm)` function to `tt-xla/python_package/tt_torch/backend/passes.py`
  that iterates the FX graph and replaces `torch.ops.prims.view_of.default` nodes
  with their input tensor
- Added import and call in `tt-xla/python_package/tt_torch/backend/backend.py`
  `torch_pass_pipeline` after `bypass_assert_tensor_metadata`

**Bug 2 — tt-forge-models loader:**
- Rewrote `_patch_clip_vision_tower_for_meta_init()` in
  `internlm_xcomposer2/pytorch/loader.py`:
  - Searches `sys.modules` for the `CLIPVisionTower` class (with `inspect.isclass` guard
    to avoid `torch.ops.CLIPVisionTower` OpNamespace)
  - Patches `load_model` to create CLIP from config with `image_size=490`
    (no `from_pretrained` inside meta context)
  - Patches `resize_pos` to no-op; returns restore thunk
- `load_model()` pre-loads the remote module via `get_class_from_dynamic_module`,
  applies the patch, calls `AutoModelForCausalLM.from_pretrained`, restores patches,
  then calls `clip_tower_cls.resize_pos(model.vit)` on the loaded model

**Bug 3 — proposed fix:**
Identify which op layer (CLIP attention, projection, or InternLM2 attention/FFN)
produces the first NaN via per-op CPU vs TT comparison. Then either:
- If a single MLIR lowering produces incorrect BF16 values: add a precision-preserving
  decomposition in `tt-mlir/lib/Conversion/StableHLOToTTIR/`
- If BF16 overflow in attention softmax: add a clamp before softmax in the
  attention decomposition

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The NaN output could arise from multiple ops across two separate sub-networks
(CLIP encoder + InternLM2 decoder). Isolating the exact op requires systematic
per-layer comparison between CPU and TT outputs, and the fix likely spans multiple
files across tt-mlir and potentially tt-xla. Additionally, one Tier A fix
(prims::view_of bypass) has already been applied to this test; the rules require
filing FAIL for any second compiler-stack bug.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    567.70s (inference ran to completion, NaN in output)
- Tier A attempts: 1 (prims::view_of bypass — successful)

## Files changed
**tt-xla (remediation/internlm_xcomposer2-pytorch-VL_7B-single_device-inference):**
- `python_package/tt_torch/backend/passes.py` — added `bypass_prims_view_of`
- `python_package/tt_torch/backend/backend.py` — import and call `bypass_prims_view_of`

**tt-forge-models (remediation/internlm_xcomposer2-pytorch-VL_7B-single_device-inference):**
- `internlm_xcomposer2/pytorch/loader.py` — reworked `_patch_clip_vision_tower_for_meta_init`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a916d0ecb8f06c29bca8c7580141033068ad46bd |
| tt-forge-models | b9c1dab8efcbeaa3b257131c33f897a10789868a |
