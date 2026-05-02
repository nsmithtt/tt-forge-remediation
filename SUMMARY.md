# Remediation Summary: orion_qwen3_5_2b_sft_v2603_v1_i1_gguf-causal_lm-pytorch-Orion_Qwen3.5_2B_SFT_v2603_v1_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[orion_qwen3_5_2b_sft_v2603_v1_i1_gguf/causal_lm/pytorch-Orion_Qwen3.5_2B_SFT_v2603_v1_i1_GGUF-single_device-inference]

## Result
NO_FIX_NEEDED — test already passes on the configured branch tip (8e215dfdf6)

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
The original CI failure was triggered before fixes were landed on the branch. The branch
`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-27` has evolved since the CI run;
by its tip commit (8e215dfdf6) all 26 GGUF loaders that install a
`_patched_load_gguf_checkpoint` module-level patch have been updated to accept
`model_to_load=None` (the kwarg added in transformers 5.2.0). Those fixes prevent the
session-contamination TypeError that was the precursor to the ImportError failure class.

The orion model loader itself has no custom GGUF patch — it relies on the canonical
`load_gguf_checkpoint` from `transformers.modeling_gguf_pytorch_utils`, which is correct.
gguf 0.18.0 is installed in the base venv, so no per-model requirements.txt is required at
runtime on this machine.

## Fix
No fix applied. The failure could not be reproduced on the configured branch. The test
passed on silicon in 414.54 s with PCC within acceptance threshold.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    414.54s (0:06:54)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8e215dfdf6bf0d76eae57455ce3e4859cb81162a |
