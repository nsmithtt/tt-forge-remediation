# Remediation Summary: airy_gguf-causal_lm-pytorch-0.8b_Q3_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[airy_gguf/causal_lm/pytorch-0.8b_Q3_K_M-single_device-inference]

## Result
FAIL — qwen35 hybrid architecture (full_attention_interval=4) cannot be loaded as qwen3 due to per-layer size mismatches; no GGUF tensor mapping exists for linear_attn layers

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-no-transformers-tensor-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

Actual failure after correct environment setup:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
Then after fixing the `_patched_load_gguf_checkpoint` signature issue:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
  model.layers.{3,7,11,15,19,23}.self_attn.q_proj.weight | MISMATCH | ckpt: torch.Size([4096, 1024]) vs model: torch.Size([1024, 1024])
  model.layers.{3,7,11,15,19,23}.self_attn.o_proj.weight | MISMATCH | ckpt: torch.Size([1024, 2048]) vs model: torch.Size([1024, 1024])
  model.layers.{3,7,11,15,19,23}.self_attn.k_proj.weight | MISMATCH | ckpt: torch.Size([512, 1024]) vs model: torch.Size([256, 1024])
  model.layers.{3,7,11,15,19,23}.self_attn.v_proj.weight | MISMATCH | ckpt: torch.Size([512, 1024]) vs model: torch.Size([256, 1024])
```

## Root cause
Three loader bugs were uncovered in sequence:

**Bug 1 (fixed): Wrong GGUF file path.**
The loader had `GGUF_FILE = "Airy-0.8b-Q3_K_M.gguf"` but the file lives in a `gguf/` subdirectory on Hugging Face: `gguf/Airy-0.8b-Q3_K_M.gguf`.

**Bug 2 (fixed): `_patched_load_gguf_checkpoint` rejected `model_to_load` kwarg.**
26 GGUF loader modules in tt_forge_models defined `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and monkey-patched this into `transformers.modeling_gguf_pytorch_utils` at module-import time (collection phase). transformers 5.2.0 added `model_to_load` to `load_gguf_checkpoint`. Since pytest imports all loaders at collection, the broken patch from any one of those loaders leaked into the Airy test's execution. Fixed by changing all 26 signatures to `(*args, **kwargs)`.

**Bug 3 (Tier B): Hybrid qwen35 architecture cannot be loaded as qwen3.**
The Airy GGUF declares `general.architecture = qwen35` with `qwen35.full_attention_interval = 4`. This means every 4th layer (indices 3, 7, 11, 15, 19, 23) is a full multi-head self-attention layer with 128 Q-heads and 16 KV-heads, while the other 22 layers are linear-attention (GLA) layers with 8 Q-heads and 2 KV-heads. The existing `_patch_qwen35_support()` converts `qwen35` → `qwen3` (model_type), which creates `Qwen3ForCausalLM`. `Qwen3Config` expects uniform head counts, so it allocates `[1024, 1024]` for all q_proj weights. The checkpoint has `[4096, 1024]` at the full-attention layers → 6 layers with mismatched sizes.

Loading as `Qwen3_5ForCausalLM` (`model_type = qwen3_5_text`) is structurally correct but fails because the gguf-py `MODEL_ARCH_NAMES` maps `'qwen35'` to `'qwen35'` (not `'qwen3_5_text'`), so the tensor-name mapping lookup raises `NotImplementedError: Unknown gguf model_type: qwen3_5_text`. The `Qwen3_5ForCausalLM` state dict has `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, etc., but the `get_tensor_name_map(MODEL_ARCH.QWEN35, n)` has no entries for those names; they fall to `perform_fallback_tensor_mapping` which is a no-op, leaving those weights randomly initialized.

## Fix
**Bugs 1 & 2** were fixed in `tt_forge_models` on branch `remediation/airy_gguf-causal_lm-pytorch-0.8b_Q3_K_M-single_device-inference`:
- `airy_gguf/causal_lm/pytorch/loader.py`: `GGUF_FILE = "gguf/Airy-0.8b-Q3_K_M.gguf"`
- 26 loader files: `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `def _patched_load_gguf_checkpoint(*args, **kwargs)` with matching call-site fix

**Bug 3 (proposed fix):** The proper fix requires two coordinated changes:
1. Extend `_patch_qwen35_support()` to also map `qwen35` config fields (especially `full_attention_interval`, `linear_num_key_heads`, `linear_num_value_heads`) to `Qwen3_5TextConfig` fields, and convert `model_type` to `qwen3_5_text` (not `qwen3`).
2. Add a custom tensor-name mapping for the linear-attention layers (`blk.N.attn_gate`, `blk.N.attn_qkv`) to `Qwen3_5ForCausalLM` state-dict keys (`model.layers.N.linear_attn.in_proj_qkv`, `model.layers.N.linear_attn.in_proj_z`, etc.), handling the full_attention_interval boundary to differentiate full-attn from linear-attn layers. This mapping must be injected into `get_gguf_hf_weights_map` (via subclassing `TensorProcessor`) or implemented as a custom loading path.

## Tier B justification
`new-infrastructure`: Properly loading the hybrid qwen35 GGUF requires a new per-layer conditional tensor-name mapping that the current gguf-py + transformers GGUF loading infrastructure does not support. The linear-attention layers (`blk.N.attn_qkv`, `blk.N.attn_gate`, etc.) have no entries in `get_tensor_name_map(MODEL_ARCH.QWEN35, n)`, and `perform_fallback_tensor_mapping` is a no-op — building the correct mapping requires understanding the GLA weight layout and injecting a custom processor that handles layer-type-conditional key translation.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/airy_gguf/causal_lm/pytorch/loader.py` — fix GGUF_FILE path
- `tt_forge_models/<26 qwen35 loaders>/causal_lm/pytorch/loader.py` — fix _patched_load_gguf_checkpoint signature

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 403ffc78fa839bd4fb3d6a1f52b63b5c5986c8b5 |
