# Remediation Summary: deepseek-deepseek_ocr-pytorch-Ocr-Unsloth-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_ocr/pytorch-Ocr-Unsloth-single_device-inference]

## Result
FAIL — Shardy propagation rejects dynamically-shaped tensor from stablehlo.set_dimension_size in the image encoder

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
stablehlo-set-dimension-size-shardy-static-shape-required

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Actual failure chain uncovered during investigation:

**Failure 1 (loader):** `AttributeError: 'ModelLoader' object has no attribute 'variant'` — fixed.

**Failure 2 (loader):** `ImportError: cannot import name 'DeepseekV2MoE'` from transformers 5.x — fixed.

**Failure 3 (loader):** `ValueError: IRD_LF_CACHE environment variable is not set` (private CI image path in load_inputs) — fixed.

**Failure 4 (loader):** `IndexError: index out of range` in `torch.embedding` caused by garbage `position_ids` buffer in `CLIPVisionEmbeddings` (transformers 5.x `init_empty_weights()` leaves buffers missing from checkpoint as uninitialized memory) — fixed.

**Failure 5 (loader):** `TypeError: DeepseekV2Experts.forward() missing 2 required positional arguments: 'top_k_index' and 'top_k_weights'` — wrong alias: `DeepseekV2MoE` should map to `DeepseekV2Moe` (lowercase), not `DeepseekV2Experts` — fixed.

**Failure 6 (tt-xla):** `NameError: name 'L' is not defined` while executing `%_guards_fn : [num_users=0] = call_module[target=_guards_fn]` — dead symbolic-shape guard nodes injected by torch.export during run_decompositions re-tracing; their generated closure references `L` (the dynamo locals dict) which is not available when PropagateUnbackedSymInts executes the graph — fixed.

**Failure 7 (tt-mlir, Tier B):**
```
loc("set-dimension-size.84"): error: Shardy propagation only supports ranked tensors
with a static shape. type: 'tensor<?x2xi32, #stablehlo.bounds<1168640, ?>>'
ValueError: Error code: 13
```

## Root cause
The model (DeepSeek-OCR Unsloth variant) processes images with dynamic sizes. The image preprocessing pipeline generates a `stablehlo.set_dimension_size` op that produces tensors with dynamically-shaped type `tensor<?x2xi32>`. The Shardy propagation pass in tt-mlir requires all tensor shapes to be statically known and rejects this dynamic type, terminating the compilation with PJRT error code 13 (INTERNAL).

## Fix
**Loader fixes (tt_forge_models):**
- `self.variant` → `self._variant` typo fix
- `DeepseekV2MoE` alias: map to `DeepseekV2Moe` (not `DeepseekV2Experts`)
- Replace private CI image path with public HuggingFace URL for test image
- Re-initialize `CLIPVisionEmbeddings.position_ids` after `from_pretrained` (transformers 5.x uninit buffer bug)

**Compiler frontend fix (tt-xla):**
In `python_package/tt_torch/backend/backend.py`, `torch_pass_pipeline`:
1. Temporarily patch `PropagateUnbackedSymInts.run_node` to return `None` for dead (num_users=0) `_guards_fn` call_module nodes, so `run_decompositions` can complete its metadata pass without `NameError`.
2. After `program.module()`, erase any surviving `_guards_fn` nodes from the compiled graph module before passes run, preventing the same error at inference time.

**For the Tier B bug (proposed):** To support `stablehlo.set_dimension_size`, tt-mlir would need to either (a) materialise dynamic dimensions into static shapes before Shardy propagation, or (b) extend Shardy to propagate through bounded-dynamic tensor types. This requires new infrastructure in tt-mlir's lowering pipeline.

## Tier B justification (FAIL with Tier=B only)
new-infrastructure — Supporting `stablehlo.set_dimension_size` through Shardy propagation requires either converting dynamic dimensions to static (requiring shape analysis at compile time) or extending Shardy to handle `#stablehlo.bounds<...>` tensor types. Neither fits in one or two files; the fix would touch multiple passes across tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    248.66s (final run showing Tier B error)
- Tier A attempts: 1 (guard node removal — effective, clears `_guards_fn` NameError)

## Files changed
- `tt-xla/python_package/tt_torch/backend/backend.py` — guard node removal (Tier A, committed)
- `tt-xla/third_party/tt_forge_models/deepseek/deepseek_ocr/pytorch/loader.py` — 5 loader fixes (committed)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d55e1666163b8753b34feeb30efe8a99372f21bf |
| tt-forge-models | f625789b0293bc03fff5fed1be7cc4171ea8dd20 |
