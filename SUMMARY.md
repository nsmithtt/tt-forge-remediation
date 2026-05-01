# Remediation Summary: llama3_1_8b_instruct_openbookqa_sft_g

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama3_1_8b_instruct_openbookqa_sft_g/pytorch-Llama3.1_8B_Instruct_OpenbookQA_SFT_G-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on branch aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-2

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
Extension modules: numpy._core._multiarray_umath, ... (total: 222)

This is a pytest subprocess crash report (`pytest-forked` format), indicating the test
process exited abnormally (SIGSEGV or similar) during a prior CI run.

## Root cause
The test passes on the current configured branch (tt-forge-models at
aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-2, commit 41abcb222a). The prior crash
was not reproducible. The loader uses `PeftModel.from_pretrained` to apply a LoRA adapter
(`qiaw99/Llama3.1-8B-Instruct-OpenbookQA-SFT-G`) on top of `unsloth/Meta-Llama-3.1-8B-Instruct`,
merges and unloads the adapter, and runs inference — all of which completes correctly on
current versions of the stack.

## Fix
No fix required. The test was re-run and passed in 258.66s (0:04:18).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    258.66s (0:04:18)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 41abcb222a (aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-2) |
