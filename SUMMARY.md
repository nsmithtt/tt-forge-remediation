# Remediation Summary: chocolatine_gguf-causal_lm-pytorch-3B_Instruct_DPO_Revised_i1_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chocolatine_gguf/causal_lm/pytorch-3B_Instruct_DPO_Revised_i1_Q4_K_M-single_device-inference]

## Result
SILICON_PASS — fix was already on branch arch-c-36-tt-xla-dev/nsmith/hf-bringup-12 at commit 5d5c309654; test passes once the branch is properly checked out at its tip

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 22:57:57.828 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)

## Root cause
The actual root cause was `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Transformers 5.x calls `load_gguf_checkpoint` with a `model_to_load` keyword argument during `from_pretrained`. Multiple GGUF loaders (26 in total) monkey-patch `transformers.modeling_utils.load_gguf_checkpoint` at import time with a narrow signature `(gguf_path, return_tensors=False)`. When pytest collects all loader modules during test discovery, the first alphabetically-imported GGUF loader (e.g., `bartowski_coniccat_qwen3_5_27b_writer_gguf`) installs its patch globally; when chocolatine's test subsequently calls `AutoModelForCausalLM.from_pretrained()`, transformers passes `model_to_load=dummy_model` to the already-patched function and gets a TypeError.

The TT_FATAL messages in the original failure log ("Chip 0 logical eth core connects to a remote mmio device") are harmless runtime warnings that the tt-metal runtime already handles by logging at WARNING level and skipping the ethernet cores. They do not cause test failure.

The fix — switching all `_patched_load_gguf_checkpoint` wrappers from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` so extra kwargs flow through to the original — was already committed to branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-12` at commit `5d5c309654` ("Forward all args in _patched_load_gguf_checkpoint wrappers", 2026-04-24). The failure was reproduced on an older detached HEAD (`0f7b734348`) before being re-run on the branch tip where it passes.

## Fix
No new code changes were needed. The fix was already present in `tt-forge-models` at commit `5d5c309654` which updated 26 GGUF loader files across the following pattern:

Before: `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):`
After:  `def _patched_load_gguf_checkpoint(*args, **kwargs):`

And the internal call to `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` became `_orig_load_gguf_checkpoint(*args, **kwargs)`.

Files changed (representative sample from commit 5d5c309654):
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- ... (26 files total)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    250.15s (0:04:10)
- Tier A attempts: N/A

## Files changed
None (fix was already on the configured branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5d5c309654dced79fe59a7ddf07390a724760f76 |
