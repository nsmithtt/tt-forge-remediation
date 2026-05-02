# Remediation Summary: phi1_5_gguf-causal_lm-pytorch-Phi_1_5_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[phi1_5_gguf/causal_lm/pytorch-Phi_1_5_Q4_K_M-single_device-inference]

## Result
SILICON_PASS — four loader bugs fixed; test passes on TT silicon in 284s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-phi2-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise NotImplementedError(
```
from `transformers/modeling_gguf_pytorch_utils.py` inside `get_gguf_hf_weights_map`:
```
NotImplementedError: Unknown gguf model_type: phi in gguf-py. This might because
you're using an outdated version of gguf-py package...
```
and, earlier in development, `ValueError: Asking to pad but the tokenizer does not have a padding token`.

## Root cause
Four distinct loader bugs, all in `tt_forge_models/phi1_5_gguf/causal_lm/pytorch/loader.py`:

1. **phi2 GGUF architecture not registered.** transformers 5.x `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES` don't include "phi2" (the GGUF architecture name for Phi-1.5/Phi-2). `GGUF_TO_FAST_CONVERTERS` also lacks a "phi2" entry. The loader now registers all three via `_patch_phi2_support()`.

2. **Narrow-sig patch-chain clobbering.** 26+ other GGUF model loaders monkey-patch `_gguf_utils.load_gguf_checkpoint` with narrow signatures that don't forward `model_to_load` (added in transformers 5.2.0). When phi1_5 is imported before these loaders and they subsequently clobber the module attribute, re-applying our patch at call time was insufficient because our captured `_orig_load_gguf_checkpoint` was itself a narrow-sig function from a loader imported before phi1_5. Fix: `_find_true_gguf_loader()` traverses `__closure__` cells and `__globals__` (via `co_names`) from `_import_time_gguf_loader` to find the original transformers function that has `model_to_load` in its explicit parameters.

3. **`get_gguf_hf_weights_map` not patched.** `load_gguf_checkpoint` remaps the GGUF result's `model_type` from `"phi2"` to `"phi"` so that `AutoConfig` and `AutoModel` resolve to the correct HF class (`PhiForCausalLM`). But the true original `load_gguf_checkpoint` then calls `get_gguf_hf_weights_map(model_to_load, processor)` which reads `hf_model.config.model_type` ("phi") and looks it up in gguf-py's `MODEL_ARCH_NAMES` which only has `"phi2"` and `"phi3"`, raising `NotImplementedError`. Fix: `_patched_get_gguf_hf_weights_map` intercepts the call and remaps `"phi"` back to `"phi2"` before calling through.

4. **`pad_token` not set after tokenizer load.** `GGUFGPTConverter` creates a fast tokenizer without propagating the GPT-2 special tokens (`eos_token`, `bos_token`). The existing `pad_token = eos_token` guard therefore assigned `None` to `pad_token`. Fix: fall back to `"<|endoftext|>"` (token 50256, always present in the GPT-2 vocabulary used by phi-1.5) when `eos_token` is also None.

## Fix
All changes in `tt_forge_models/phi1_5_gguf/causal_lm/pytorch/loader.py` on branch
`remediation/phi1_5_gguf-causal_lm-pytorch-Phi_1_5_Q4_K_M-single_device-inference`.

- Added `_find_true_gguf_loader()`: DFS traversal through `__closure__` and `__globals__` to find the transformers-original `load_gguf_checkpoint` (identified by `model_to_load` in explicit params).
- Restructured into `_apply_patches()` called both at import time and at the start of every public method to re-apply after later-loader clobbering.
- Added `_patched_get_gguf_hf_weights_map()`: remaps `model_type="phi"` to `"phi2"` before the gguf-py architecture lookup.
- Fixed `pad_token` fallback: `eos_token or "<|endoftext|>"`.

Two commits:
- `0e4355a1ad` — initial phi2 registration and model_type remap
- `e6735c84b3` — bypass narrow-sig patch chain, patch get_gguf_hf_weights_map, fix pad_token fallback

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    284.24s (0:04:44)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/phi1_5_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e6735c84b3c2a7eafbff3c3f91e9e558d198ca7f |
