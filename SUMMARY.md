# Remediation Summary: progenitor_v3_2_llama_70b_gguf-causal_lm-pytorch-V3.2_LLaMa_70B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[progenitor_v3_2_llama_70b_gguf/causal_lm/pytorch-V3.2_LLaMa_70B_i1_GGUF-single_device-inference]

## Result
XFAIL — Progenitor V3.2 LLaMa 70B Q4_K_M GGUF (42.5 GB) exceeds p150b single-device DRAM (32 GB)

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
Test exceeded configured timeout and was killed

## Root cause
The model `mradermacher/Progenitor-V3.2-LLaMa-70B-i1-GGUF` uses the GGUF
file `Progenitor-V3.2-LLaMa-70B.i1-Q4_K_M.gguf`, which is 42,520,398,848
bytes (~42.5 GB) on disk. The single p150b device has 32 GB of DRAM. Even at
Q4_K_M quantization (~4.5 bits/weight), the 70B parameter model cannot fit in
device memory.

The CI failure manifests as "Test exceeded configured timeout and was killed"
because:
1. Downloading the 42.5 GB GGUF file from HuggingFace (not cached) takes
   ~15-20 minutes on the CI runner.
2. Attempting to load and tilize the weights into device DRAM then OOMs.
3. The total run time exceeds the CI-configured timeout before the OOM error
   can be surfaced.

This is identical in class to BrownLoafers-70B-i1-GGUF and
Llama-MiraiFanfare-2-3.3-70B-i1-GGUF — both 70B LLaMA Q4_K_M GGUF models
that fail with the same hardware capacity ceiling on p150b.

The loader already has `gguf>=0.10.0` in requirements.txt; no loader fix is
needed. This is a pure hardware capacity limitation.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry for this test to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in
the tt-xla repo (remediation branch
`remediation/progenitor_v3_2_llama_70b_gguf-causal_lm-pytorch-V3.2_LLaMa_70B_i1_GGUF-single_device-inference`,
commit `30170de6b11049d5303f3041882841e91438cd3d`).

No changes to tt-mlir or tt-metal were required.

## Verification
- pytest exit: TIMEOUT (hardware capacity; not run to completion)
- Hardware:    blackhole-p150b
- Duration:    not-run (model not in HF cache; full run would OOM)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 30170de6b11049d5303f3041882841e91438cd3d |
| tt-forge-models | 268b8acf15a297a1f2baf60645ee263bd0787057 |
