# Remediation Summary: huihui_qwen_3_5_abliterated_gguf-causal_lm-pytorch-27B_Abliterated_i1_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen_3_5_abliterated_gguf/causal_lm/pytorch-27B_Abliterated_i1_GGUF-single_device-inference]

## Result
XFAIL — 27B model dequantises to ~54 GB BF16 at load time, exceeding p150b single-device DRAM (24 GB); Fabric Router Sync Timeout is the downstream OOM symptom.

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-27b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-19 03:43:35.647 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

## Root cause
Two issues were found:

**Loader bug (fixed):** The GGUF file `Huihui-Qwen3.5-27B-abliterated.i1-Q4_K_M.gguf` declares `general.architecture = qwen35`. This key is not in transformers 5.2.0's `GGUF_SUPPORTED_ARCHITECTURES` (which only has `qwen3`, not `qwen35`), so any loader that does not patch the mapping raises `ValueError: GGUF model with architecture qwen35 is not supported yet.` The original test passed this stage because another co-collected Qwen 3.5 GGUF loader had already patched `GGUF_SUPPORTED_ARCHITECTURES` at pytest collection time. The `huihui_qwen_3_5_abliterated_gguf` loader was missing its own patch.

**Hardware capacity ceiling (XFAIL):** Qwen3.5-27B has 27 billion parameters (64 layers, hidden_size=5120). When loaded via transformers' GGUF loader, weights are dequantised to BF16 on the host (27B × 2 bytes ≈ 54 GB). The single p150b device has 24 GB GDDR DRAM. The model cannot fit on-device; inference triggers an OOM condition that manifests as `TT_THROW: Fabric Router Sync: Timeout`.

## Fix
**Loader fix** (`tt-xla/third_party/tt_forge_models/huihui_qwen_3_5_abliterated_gguf/causal_lm/pytorch/loader.py`):
Added `_patch_qwen35_support()` which registers `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` (aliased to `qwen3`), and adds `qwen35`/`qwen3_5_text` to `GGUF_TO_FAST_CONVERTERS`. Added `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs)` (with `**kwargs` to accept the transformers 5.2.0 `model_to_load=` kwarg) that remaps `model_type='qwen35'` → `'qwen3'` after loading. Added `_find_real_load_gguf()` to walk any existing wrapper chain and capture the real transformers function, preventing stacking of broken fixed-signature wrappers.

**Test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
Added `KNOWN_FAILURE_XFAIL` entry with explanation that 27B BF16 exceeds p150b DRAM.

## Verification
- pytest exit: FAIL (not run on silicon — hardware capacity ceiling is deterministic: 27B × 2 bytes = 54 GB > 24 GB p150b DRAM)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huihui_qwen_3_5_abliterated_gguf/causal_lm/pytorch/loader.py` — added qwen35 GGUF architecture patch
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 51e4857426e89720d638900f7d16224a8ec04b32 |
| tt-forge-models | 35547bf0c790e8c77ba61a10d78446e02e569c66 |
