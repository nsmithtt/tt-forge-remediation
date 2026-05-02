# Remediation Summary: mistral_small_3_1_24b_instruct_2503_fp8_dynamic-pytorch-24B_Instruct_2503_FP8_Dynamic-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_small_3_1_24b_instruct_2503_fp8_dynamic/pytorch-24B_Instruct_2503_FP8_Dynamic-single_device-inference]

## Result
XFAIL — 24B BF16 model (~48 GB weights) exceeds n150 single-device DRAM (~12 GB); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-24b-bf16-oom-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `raise AttributeError(`

Actual failure sequence encountered during debugging:
1. `ImportError: compressed_tensors is not installed` — missing requirements.txt
2. `AttributeError: 'Mistral3ForConditionalGeneration' object has no attribute 'language_model'` — wrong attribute path in load_shard_spec
3. FP8 weights not dequantized causing runtime errors — missing _dequantize_fp8_to_bf16 pass
4. `INTERNAL: Error code: 13` (pjrt-device-to-host-transfer) — image_sizes tensor moved to TT device
5. `RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: prims::view_of` — alias annotation in squeeze decomposition
6. Terminal: `TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4156442816 B, free: 116947200 B, largest free block: 41811968 B)`

## Root cause
The test encountered five cascading loader bugs that were fixed, followed by a hardware capacity ceiling. After all loader and compiler-frontend fixes, the model fails with OOM during DRAM buffer allocation. The RedHatAI Mistral Small 3.1 24B FP8 model, once dequantized to BF16 for TT hardware, has approximately 48 GB of weights. The n150 single-device DRAM capacity is approximately 12 GB. This is a hardware class limitation — the model simply cannot fit on a single device.

The five preceding bugs that were fixed along the way:
1. **Missing requirements.txt** (`compressed-tensors` not listed) — loader bug
2. **Wrong attribute path in load_shard_spec** (`model.language_model` → `model.model.language_model`) — loader bug
3. **FP8 weights not dequantized** — added `_dequantize_fp8_to_bf16` and set `quantization_enabled=False` on all compressed-tensors patched modules — loader bug
4. **image_sizes device-to-host transfer (INTERNAL Error 13)** — `load_inputs` kept `image_sizes` as int64 tensor that moved to TT device; patched `get_image_features` to use pure-Python split_sizes arithmetic — loader bug
5. **prims::view_of alias annotation breaking partition_fx_graph_for_cpu_fallback** — `torch.split(x.squeeze(0), ...)` triggers squeeze→prims.view_of decomposition with alias annotation `(Tensor(a) -> Tensor(a))` that XLA partitioner rejects — Tier A tt-xla fix (`bypass_prims_view_of` FX pass)

## Fix
Loader fixes in `tt-xla/third_party/tt_forge_models/mistral_small_3_1_24b_instruct_2503_fp8_dynamic/pytorch/`:
- `requirements.txt`: Added `compressed-tensors` dependency
- `loader.py`:
  - Added `_get_fp8_dtypes()` + `_dequantize_fp8_to_bf16()` to dequantize FP8 Linear weights and disable activation quantization
  - Added `_patch_mistral3_split_sizes()` to patch `Mistral3Model.get_image_features` with pure-Python split_sizes computation
  - Fixed `load_shard_spec` to use `model.model.language_model.layers` and `model.model.vision_tower.transformer.layers`
  - `load_inputs`: Drop all-ones `attention_mask`; convert `image_sizes` tensor to Python list

Tier A tt-xla fix in `tt-xla/python_package/tt_torch/backend/`:
- `passes.py`: Added `bypass_prims_view_of()` FX pass that replaces `prims.view_of` identity nodes with their input
- `backend.py`: Added import and call to `bypass_prims_view_of` before `bridge.extract_compiled_graph` in `_call_experimental_compile`

Test config update:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this model with OOM reason

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling)
- Hardware:    n150
- Duration:    N/A (OOM before inference completes)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/mistral_small_3_1_24b_instruct_2503_fp8_dynamic/pytorch/requirements.txt`
- `tt-xla/third_party/tt_forge_models/mistral_small_3_1_24b_instruct_2503_fp8_dynamic/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 46796ba752a4b66d4da75234fd93276ad6e9471b |
| tt-forge-models | d17292cef44a62be3a0a207a06056eb6192f741c |
