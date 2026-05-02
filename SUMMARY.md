# Remediation Summary: nanonets_ocr2_aio_gguf-image_to_text-pytorch-3B_AIO_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nanonets_ocr2_aio_gguf/image_to_text/pytorch-3B_AIO_GGUF-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer Tier B: VLM vision encoder calls .tolist() on TT device tensors for Python control flow; TT PJRT rejects D2H transfers in compiled graph context

## Stack layer
loader, tt-xla

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

The vision encoder's `rot_pos_emb`, `get_window_index`, `get_rope_index`, and
`get_image_features` methods call `.tolist()` on `grid_thw` / `input_ids` /
`image_grid_thw` tensors for Python control flow (loop bounds, index offsets).
At runtime these tensors are on the TT device and `.tolist()` requires an
eager D2H transfer that TT PJRT does not support.

Original stated failure:
```
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast
processor by default, even if the model checkpoint was saved with a slow
processor. This is a breaking change and may produce slightly different
outputs. To continue using the slow processor, instantiate this class with
`use_fast=False`.
```

## Root cause
The loader had five bugs, all fixed:

1. **GGUF arch not registered**: `qwen2vl` was missing from
   `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` in
   transformers 5.x, causing `load_gguf_checkpoint` to refuse the file.

2. **model_type mismatch**: `get_gguf_hf_weights_map` reads
   `model.config.model_type = "qwen2_5_vl"` but gguf-py `MODEL_ARCH_NAMES`
   only has enum 28 = `"qwen2vl"`. Weight mapping failed with `KeyError`.

3. **use_fast=False missing**: GGUF repo (`prithivMLmods/Nanonets-OCR2-3B-AIO-GGUF`)
   has no `preprocessor_config.json`, so the processor must be loaded from the
   base model (`nanonets/Nanonets-OCR2-3B`) with `use_fast=False` to avoid the
   transformers 5.x fast-processor default breaking change.

4. **Incomplete GGUF config**: GGUF file's metadata gives an incomplete
   `qwen2_5_vl` config (no vision sub-config). Fix: load full config from base
   model and pass it explicitly to `from_pretrained`.

5. **Narrow-sig `load_gguf_checkpoint` patch**: 26 other loaders patched
   `_orig_load_gguf_checkpoint` with a fixed 2-arg signature, dropping the
   `model_to_load` kwarg added in transformers 5.2.0. The session-level
   monkey-patch was silently broken when nanonets ran after any of those loaders.
   Fix: widen all 26 patches to `(*args, **kwargs)`.

After all loader fixes, the model loads and starts compiling. The terminal
failure is in the TT PJRT layer: `Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb`
calls `grid_thw.tolist()` to compute RoPE index offsets. This is normal Python
control flow, not a traced computation, but Dynamo has captured `grid_thw` as
an XLA tensor. When `extract_compiled_graph_helper` calls `torch_xla.sync()` to
materialize the compiled graph, PJRT attempts a D2H transfer and rejects it with
INTERNAL: Error code: 13. The same path is triggered by `get_window_index`,
`get_rope_index`, and `get_image_features`.

Wrapping each call-site with `.cpu()` was attempted but creates explicit D2H
ops in the Dynamo-traced graph, triggering the same INTERNAL:13 at compile
time rather than run time. The correct fix is in the TT PJRT layer: implement
D2H tensor transfer in the compiled graph context (PJRT `TransferToInfeed` /
`BufferToHost` path) so that graph-break reads on small integer metadata
tensors succeed.

## Fix
Loader-layer fixes committed to
`remediation/nanonets_ocr2_aio_gguf-image_to_text-pytorch-3B_AIO_GGUF-single_device-inference`
in tt-forge-models (8bfb3267c0):

- `nanonets_ocr2_aio_gguf/image_to_text/pytorch/loader.py`:
  - Registered `qwen2vl` in `GGUF_SUPPORTED_ARCHITECTURES` and
    `GGUF_TO_TRANSFORMERS_MAPPING` (Bug 1)
  - Added `_patched_get_gguf_hf_weights_map` to translate `qwen2_5_vl` /
    `qwen2_vl` → `qwen2vl` for gguf-py lookup (Bug 2)
  - Load processor from `nanonets/Nanonets-OCR2-3B` with `use_fast=False`
  - Load full config from `nanonets/Nanonets-OCR2-3B` and pass to
    `from_pretrained` to bypass GGUF's incomplete config
- `nanonets_ocr2_aio_gguf/requirements.txt`: added `gguf>=0.10.0`
- 26 other loaders: widened `_patched_load_gguf_checkpoint` signature from
  `(gguf_path, return_tensors=False)` to `(*args, **kwargs)`

Terminal Tier B bug: implement D2H tensor transfer support in TT PJRT
(`tt-xla` PJRT plugin, `src/tt/pjrt_plugin.cc` / PJRT transfer path) so
that compiled graphs can perform eager reads of small integer metadata
tensors (like `image_grid_thw`) required by Python control flow in
Qwen2.5-VL's vision encoder.

## Tier B justification
**Indicator: new-infrastructure**

TT PJRT currently has no path for D2H buffer transfers inside a compiled
XLA graph context. Implementing `BufferToHost` / `TransferToInfeed` for
the compiled-graph execution path requires new infrastructure in the PJRT
plugin, coordinated between `tt-xla` and `tt-metal` transport layers.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    345.61s (0:05:45)
- Tier A attempts: N/A

## Files changed
tt-forge-models:
- `nanonets_ocr2_aio_gguf/image_to_text/pytorch/loader.py`
- `nanonets_ocr2_aio_gguf/requirements.txt`
- 26 other loader files (narrow-sig patch fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8bfb3267c0 |
