# Remediation Summary: llama_miraifanfare_2_3_3_70b_i1_gguf-causal_lm-pytorch-70B_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_miraifanfare_2_3_3_70b_i1_gguf/causal_lm/pytorch-70B_I1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 70B LLaMA Q4_K_M GGUF requires ~141 GB BF16 / ~40 GB Q4_K_M; p150b has 32 GB DRAM, OOM after allocating ~30.7 GB

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
llama-70b-q4km-dram-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks,
where each bank needs to store 58720256 B, but bank size is 4273390016 B
(allocated: 4113318208 B, free: 160071808 B, largest free block: 45351360 B)
```

## Root cause
The original CI failure ("Test exceeded configured timeout and was killed") was caused by
two issues:

1. **Loader bug (fixed on hf-bringup-18 branch)**: `gguf>=0.10.0` was missing from
   `requirements.txt`. Without it, the GGUF model file cannot be loaded by transformers,
   causing a silent hang or timeout. Fixed in commit `7083d8d6f1` on tt_forge_models branch
   `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-18`.

2. **Hardware capacity ceiling**: After the loader fix, the test runs to completion but
   crashes with a device DRAM OOM during input tensor preparation (tilize). The p150b
   Blackhole device has ~34.2 GB DRAM (8 banks × ~4.27 GB). After loading ~30.7 GB of
   model weight tensors, the runtime cannot allocate the next 469 MB buffer because
   fragmentation leaves only 45 MB of contiguous space per bank. A 70B LLaMA model
   requires ~141 GB at BF16 (the dtype used by the test runner), far exceeding any
   single-device DRAM. Even at the quantized Q4_K_M storage size (~40 GB), the model
   exceeds single-device capacity. The CI timeout was triggered because the full run
   takes ~25 minutes before hitting the OOM.

## Fix
- **Loader fix** (already applied): Added `gguf>=0.10.0` to
  `llama_miraifanfare_2_3_3_70b_i1_gguf/causal_lm/pytorch/requirements.txt` in
  tt_forge_models (commit `7083d8d6f1`).
- **Test config XFAIL**: Added `KNOWN_FAILURE_XFAIL` entry to
  `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla
  (commit `819e46d5f66ccc4ad718a1138648a7a79629ae48`).

## Verification
- pytest exit: FAIL (OOM TT_FATAL)
- Hardware:    blackhole-p150b
- Duration:    1483.87s (0:24:43)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry
- `tt-xla/third_party/tt_forge_models/llama_miraifanfare_2_3_3_70b_i1_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0` (prior Claude session, commit 7083d8d6f1)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 819e46d5f66ccc4ad718a1138648a7a79629ae48 |
| tt-forge-models | a9bfb53a4ebed412213b21f4939972372edf6dfa |
