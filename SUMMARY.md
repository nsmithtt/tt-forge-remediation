# Remediation Summary: llama_4_scout_gguf-causal_lm-pytorch-17B_16E_Instruct_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[llama_4_scout_gguf/causal_lm/pytorch-17B_16E_Instruct_GGUF-single_device-inference]

## Result
XFAIL — Llama 4 Scout (109B total MoE parameters) cannot fit on any single TT device; even the most aggressive quantization (IQ1, ~27 GB) exceeds max single-device DRAM (24 GB on p150b)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-llama4-scout-109b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: TT_THROW @ /home/nsmith/hf-bringup/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception

## Root cause
Llama 4 Scout 17B-16E is a Mixture-of-Experts model with ~109B total parameters across 16 experts (17B active parameters per inference step). When loaded as GGUF, the smallest available quantization (UD-TQ1_0) is 27.25 GB, and the originally-targeted Q4_K_M totals ~61 GB across two shards. The maximum single TT device DRAM is 24 GB (p150b). There is no quantization of this model that can fit on a single TT device.

Additionally, the loader contained a stale GGUF filename: unsloth reorganized the Q4_K_M variant from a flat file (`Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf`, no longer exists) into a sharded subdirectory layout (`Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf`). This caused an OSError locally, while the CI machine reproduced the original TT runtime error because it had the file cached from before the reorganization.

## Fix
Two changes in tt-xla remediation branch `remediation/llama_4_scout_gguf-causal_lm-pytorch-17B_16E_Instruct_GGUF-single_device-inference`:

1. **tt_forge_models loader fix** (`llama_4_scout_gguf/causal_lm/pytorch/loader.py`): Updated `GGUF_FILE` from `"Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf"` (stale flat-file path) to `"Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf"` (current sharded path in the unsloth HuggingFace repo).

2. **Test config XFAIL** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`): Added `KNOWN_FAILURE_XFAIL` entry for `llama_4_scout_gguf/causal_lm/pytorch-17B_16E_Instruct_GGUF-single_device-inference` with explanation that the model exceeds single-device DRAM.

## Verification
- pytest exit: FAIL (OSError on wrong GGUF filename; hardware-class prevents silicon run)
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llama_4_scout_gguf/causal_lm/pytorch/loader.py` — fix GGUF_FILE path
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add KNOWN_FAILURE_XFAIL

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f500bb26a9fbd52a778bac75bf8cbb784acbb427 |
| tt-forge-models | 5ff13e0a04db321b492ef06dca3cd146af37cf17 |
