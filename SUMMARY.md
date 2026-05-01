# Remediation Summary: latentdream_exp_delta_8b_gguf-causal_lm-pytorch-LatentDream_Exp_Delta_8B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[latentdream_exp_delta_8b_gguf/causal_lm/pytorch-LatentDream_Exp_Delta_8B_GGUF-single_device-inference]

## Result
SILICON_PASS

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
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Cross-loader clobbering of `load_gguf_checkpoint` in the transformers modules.
The `latentdream_exp_delta_8b_gguf` loader has no own GGUF checkpoint patch, but
26+ other GGUF loaders install narrow-sig `_patched_load_gguf_checkpoint(gguf_path,
return_tensors=False)` wrappers at module import time onto
`transformers.modeling_gguf_pytorch_utils`, `transformers.configuration_utils`, etc.
`transformers.modeling_utils.from_pretrained` (line 4016) does a local import
`from .modeling_gguf_pytorch_utils import load_gguf_checkpoint` at call time, which
picks up whatever is currently installed.  Since transformers 5.2.0 added
`model_to_load` as a third positional-or-keyword argument, the last narrow-sig
patcher to run before this test causes `TypeError: got an unexpected keyword argument
'model_to_load'`.

The chain is complicated by `onion008_qwen3_5_35b_a3b_claude_4_6_opus_reasoning_distilled_i1_gguf`
which uses a closure variable (`orig_load`) rather than `__globals__['_orig_load_gguf_checkpoint']`,
and by non-deterministic (inode-order) model import sequences from `os.walk`.  A simple
DFS via `__globals__` fails to follow these closure links.

## Fix
`latentdream_exp_delta_8b_gguf/causal_lm/pytorch/loader.py` in `tt-forge-models`.

Added two functions at module level:

1. `_find_true_orig_load_gguf()`: BFS through the entire patch chain following both
   `__globals__['_orig_load_gguf_checkpoint']` and all callable closure variables, until
   it finds the function that has `model_to_load` as an explicit named parameter (the
   true original from transformers).

2. `_restore_wide_sig_load_gguf()`: called at module import time AND inside `load_model()`
   just before `AutoModelForCausalLM.from_pretrained()`.  Checks whether the current
   `load_gguf_checkpoint` accepts `model_to_load`; if not, calls `_find_true_orig_load_gguf`
   and re-installs a `_wide(*args, **kwargs)` wrapper backed by the true original in all
   four patched modules.  Calling it twice (import time + call time) handles patchers
   installed either before or after this module's own import.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    465.50s (0:07:45)
- Tier A attempts: N/A

## Files changed
- `latentdream_exp_delta_8b_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 14918f9cf5a7f9f6d5809b89d3ab5ab91c9a6084 |
