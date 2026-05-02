# Remediation Summary: megamind_v2_vl_high_i1_gguf-image_to_text-pytorch-v2_vl_high_i1_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[megamind_v2_vl_high_i1_gguf/image_to_text/pytorch-v2_vl_high_i1_gguf-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer: grid_thw.tolist() on TT tensor in Qwen3VL fast_pos_embed_interpolate (Tier B new-infrastructure)

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
The Qwen3VL vision encoder calls `fast_pos_embed_interpolate` during the forward pass, which calls `grid_thw.tolist()` on a tensor that is on the TT device (`transformers/models/qwen3_vl/modeling_qwen3_vl.py:699`). The PJRT backend cannot transfer arbitrary tensor data from TT device to host for Python list conversion — it raises `INTERNAL: Error code: 13`.

The loader had two prior bugs that were fixed: (1) the GGUF architecture name `qwen3vl` was not registered in `GGUF_CONFIG_MAPPING`/`GGUF_SUPPORTED_ARCHITECTURES`, causing a `ValueError: GGUF model with architecture qwen3vl is not supported yet`; and (2) multiple other loaders had installed broken `load_gguf_checkpoint` wrappers with fixed signatures that drop the `model_to_load` kwarg added in transformers 5.2.0. Both loader bugs are fixed in the remediation commit. After those fixes, the terminal failure is `pjrt-device-to-host-transfer` in `fast_pos_embed_interpolate`.

## Fix
**Loader fix (committed):** `tt_forge_models/megamind_v2_vl_high_i1_gguf/image_to_text/pytorch/loader.py`
- Added `_register_qwen3vl_gguf_architecture()`: registers `"qwen3vl"` in `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`; patches `get_gguf_hf_weights_map` to remap `model_type "qwen3_vl"` → `"qwen3vl"`; installs a properly-signed `load_gguf_checkpoint` wrapper that routes `model_to_load` to the real transformers function, bypassing broken fixed-signature wrappers from other loaders
- Loads `Qwen3VLConfig.from_pretrained(BASE_MODEL)` explicitly to avoid GGUF flat-field config misparse
- Sets `ignore_mismatched_sizes=True`
- Sets pixel limits (`min_pixels=56*56`, `max_pixels=13*28*1280`) on the image processor

**Terminal compiler-stack bug (not fixed):** The `pjrt-device-to-host-transfer` bug lives in the tt-xla PJRT backend. `tensor.tolist()` and `tensor.item()` on TT-device tensors require device→host data transfer, which the current PJRT implementation does not support for arbitrary tensors during compilation/execution. A fix would require implementing the tensor data transfer path in the PJRT backend — new infrastructure.

## Tier B justification
new-infrastructure — implementing device→host tensor data transfer for `.tolist()`/`.item()` calls in the PJRT backend requires new infrastructure in the tt-xla PJRT plugin and/or tt-metal runtime. No scoped single-function fix exists.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    251.55s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/megamind_v2_vl_high_i1_gguf/image_to_text/pytorch/loader.py` (loader fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 98d160996e8f373981e29f12561690c8f610c3bc |
| tt-forge-models | 02c5387b368d7bee3e7b0af6cd04c51ee914ebc4 |
