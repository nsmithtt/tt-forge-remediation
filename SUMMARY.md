# Remediation Summary: ice0_64_24_01_rp_i1_gguf-pytorch-Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ice0_64_24_01_rp_i1_gguf/causal_lm/pytorch-Ice0_64_24_01_RP_i1_Q4_K_M_GGUF-single_device-inference]

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
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

In full pytest sessions the immediate error was:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
Two bugs in the loader layer:

1. **Missing requirements.txt**: The loader directory had no `requirements.txt` with
   `gguf>=0.10.0`, so fresh environments raised ImportError before even reaching the
   model load.

2. **Global patcher chain not walked**: In a full pytest session, other loaders (e.g.
   `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`,
   `qwen_3_5_35b_a3b_claude_opus_reasoning_gguf`) patch
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with
   functions that lack the `model_to_load` and `torch_dtype` kwargs that transformers 5.x
   always passes.

   The patcher chain nests in two distinct patterns: some patchers save the previous
   function as a module-level name (visible in `__globals__`), while others capture it in
   a closure (`__closure__`). The initial fix attempt only walked one level of globals,
   missing chains where an intermediate patcher used a closure (qwen35) or where the
   first-level saved function was yet another patcher (bartowski saves tvall43's patcher
   in its globals, not the real transformers function).

   The correct fix is a BFS through all gguf-named callables reachable via `__globals__`
   and `__closure__` of each patcher in the chain, until the function from
   `transformers.modeling_gguf_pytorch_utils` with `model_to_load` in its signature is
   found. The context manager then temporarily installs this real function, runs
   `from_pretrained`, and restores the patcher chain.

## Fix
**File 1**: `ice0_64_24_01_rp_i1_gguf/causal_lm/pytorch/requirements.txt` (created)
- Added `gguf>=0.10.0` to ensure the gguf library is always installed.

**File 2**: `ice0_64_24_01_rp_i1_gguf/causal_lm/pytorch/loader.py`
- Added `_find_original_from_transformers(fn)`: BFS through `__globals__` and `__closure__`
  of the outermost patched function, following only callables whose name or key contains
  "gguf" or "checkpoint" (to avoid traversing all of Python's stdlib). Returns the first
  callable whose `__module__` is `transformers.modeling_gguf_pytorch_utils` and whose
  signature includes `model_to_load`.
- Updated `_gguf_kwargs_compat()` context manager to call `_find_original_from_transformers`
  and temporarily install the real function for the duration of `from_pretrained`.

Both fixes are in `tenstorrent/tt-forge-models` on branch
`remediation/ice0_64_24_01_rp_i1_gguf-pytorch-Q4_K_M_GGUF-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    294.65s (0:04:54)
- Tier A attempts: N/A

## Files changed
- `ice0_64_24_01_rp_i1_gguf/causal_lm/pytorch/requirements.txt` (created)
- `ice0_64_24_01_rp_i1_gguf/causal_lm/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 14e5d4a94d0f896b1b9658f835f09f602ca07f8d |
| tt-forge-models | 4930b9516180552967e6299ac18b17644d93b1af |
