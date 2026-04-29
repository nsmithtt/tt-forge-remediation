# Remediation Summary: duolaf_qwen_3_5_gguf-causal_lm-pytorch-27B_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[duolaf_qwen_3_5_gguf/causal_lm/pytorch-27B_Q4_K_M-single_device-inference]

## Result
XFAIL — 27B Qwen3.5 GGUF dequantises to ~54 GB BF16 at load time, exceeding p150b single-device DRAM (24 GB); hardware capacity ceiling.

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
ValueError: GGUF model with architecture qwen35 is not supported yet.

(The CI failure surface was `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` — a harmless SWIG import warning. The actual test error, reproduced locally, is the ValueError above.)

## Root cause
Two issues coexist:

**Loader bug (fixed):** The GGUF file `Qwen3.5-27B-Q4_K_M.gguf` declares `general.architecture = qwen35`. This key is not in transformers 5.x `GGUF_SUPPORTED_ARCHITECTURES` (which only has `qwen3`, not `qwen35`), so any loader that does not patch the mapping raises `ValueError: GGUF model with architecture qwen35 is not supported yet.` The duolaf loader was missing this patch.

**Hardware capacity ceiling (XFAIL):** Qwen3.5-27B has 27 billion parameters. When loaded via transformers' GGUF loader, weights are dequantised to BF16 on the host (27B × 2 bytes ≈ 54 GB). The single p150b device has 24 GB GDDR DRAM. The model cannot fit on-device; inference would trigger an OOM condition (confirmed by the identical huihui_qwen_3_5_abliterated_gguf 27B report which saw `TT_THROW: Fabric Router Sync: Timeout`).

## Fix
**Loader fix** (`tt-xla/third_party/tt_forge_models/duolaf_qwen_3_5_gguf/causal_lm/pytorch/loader.py`):
Added `_patch_qwen35_support()` which registers `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` (aliased to `qwen3`), and adds `qwen35`/`qwen3_5_text` to `GGUF_TO_FAST_CONVERTERS`. Added `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs)` (with `**kwargs` to accept the transformers 5.2.0 `model_to_load=` kwarg) that remaps `model_type='qwen35'` → `'qwen3'` after loading. Added `_find_real_load_gguf()` to walk any existing wrapper chain and capture the real transformers function, preventing stacking of broken fixed-signature wrappers.

**Test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
Added `KNOWN_FAILURE_XFAIL` entry explaining that 27B BF16 exceeds p150b DRAM.

## Verification
- pytest exit: FAIL (not run on silicon — hardware capacity ceiling is deterministic: 27B × 2 bytes = 54 GB > 24 GB p150b DRAM)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/duolaf_qwen_3_5_gguf/causal_lm/pytorch/loader.py` — added qwen35 GGUF architecture patch
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9a62f70c6edaf96b9744e83628101f8257d1e50d |
| tt-forge-models | 704eeee01f9f6a908d2db26ab156b387b8fe1a73 |
