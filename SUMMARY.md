# Remediation Summary: hermes_4_70b_heretic_i1_gguf-causal_lm-pytorch-70B_heretic_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hermes_4_70b_heretic_i1_gguf/causal_lm/pytorch-70B_heretic_i1_GGUF-single_device-inference]

## Result
XFAIL — 70B LLaMA model dequantizes to ~141 GB BF16, exceeding n150 single-device DRAM capacity (~32 GB); CI hangs at ~90% of GGUF dequantization due to host RAM exhaustion.

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure (on hf-bringup-17 branch, after loader fixes already applied):
```
Converting and de-quantizing GGUF tensors...:  90%|████████▉ | 648/724 [05:18<00:44,  1.71it/s]
```
The process hung at 90% of GGUF weight dequantization and did not complete.

## Root cause
Two issues identified:

**Loader fixes (already applied on hf-bringup-17 before CI run):**
1. `requirements.txt` with `gguf>=0.10.0` was missing — added in commit `8fefb0d92a`.
2. After dynamic install by `RequirementsManager`, `transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING` is stale (computed at import time), causing `is_gguf_available()` to return version `'N/A'` and crash with `packaging.version.InvalidVersion`. Fixed by refreshing the mapping and clearing the LRU cache in `_refresh_gguf_detection()` before any GGUF call — commit `3ae9fcc580`.

**Hardware capacity (root cause of the CI hang):**
The model `mradermacher/Hermes-4-70B-heretic-i1-GGUF` (`Hermes-4-70B-heretic.i1-Q4_K_M.gguf`) is a LLaMA 3.1-based 70B model (80 layers, hidden_size=8192, intermediate_size=28672, vocab_size=128256). When loaded by `AutoModelForCausalLM.from_pretrained(gguf_file=...)`, all 724 GGUF tensors are dequantized from Q4_K_M to BF16 in CPU RAM. The resulting in-memory model requires ~141 GB of CPU RAM. The CI machine exhausted available RAM at ~648/724 tensors (~89.5%, corresponding to roughly layer 72 of 80), causing the process to hang (OOM-killed or paged to swap). Even if loading were to complete, the n150 single-device DRAM (~32 GB across 8 banks of ~4 GB each) cannot accommodate the dequantized 141 GB model.

## Fix
**Loader fixes (in tt_forge_models, already on hf-bringup-17):**
- `hermes_4_70b_heretic_i1_gguf/causal_lm/pytorch/requirements.txt`: added `gguf>=0.10.0` (commit `8fefb0d92a`)
- `hermes_4_70b_heretic_i1_gguf/causal_lm/pytorch/loader.py`: added `_refresh_gguf_detection()` called before tokenizer/model load, refreshing `PACKAGE_DISTRIBUTION_MAPPING` and clearing `is_gguf_available` LRU cache (commit `3ae9fcc580`)

**Test config (tt-xla remediation branch):**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: added `KNOWN_FAILURE_XFAIL` entry explaining the 70B hardware capacity ceiling

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: not-run
- Hardware:    n150
- Duration:    n/a (model not cached locally; 40 GB GGUF download + ~141 GB BF16 dequantization would OOM n150 DRAM regardless)
- Tier A attempts: N/A

## Files changed
- `hermes_4_70b_heretic_i1_gguf/causal_lm/pytorch/requirements.txt` (tt_forge_models, pre-existing fix on hf-bringup-17)
- `hermes_4_70b_heretic_i1_gguf/causal_lm/pytorch/loader.py` (tt_forge_models, pre-existing fix on hf-bringup-17)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 694d1136fb66bf4cd8b4cf236c63a9d40f533bfe |
| tt-forge-models | 3ae9fcc58018e0ac84d934c407a7b23cfff8d8c7 |
