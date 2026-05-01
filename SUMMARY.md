# Remediation Summary: mindbot_ultra_4b_16b_i1_gguf-causal_lm-pytorch-4B_16b_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mindbot_ultra_4b_16b_i1_gguf/causal_lm/pytorch-4B_16b_i1_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5 GatedDeltaNet scan-loop unrolls into huge StableHLO scatter graph; TT MLIR compiler hangs indefinitely during compilation

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ssm-scan-loop-compilation-hang

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing the loader: test loads, initializes device, compiles multiple small modules (embedding, norm), then hangs for 12+ minutes in the main transformer compilation with no output or progress.

## Root cause
Two loader bugs and one compiler-stack bug were found in sequence:

**Bug 1 (loader)**: Other loaders (e.g. `gpt_oss_swallow_120b_rl_v0_1_gguf`) globally patch `load_gguf_checkpoint` at import time with a signature that omits `model_to_load`. Transformers 5.x passes `model_to_load=dummy_model` when calling `load_gguf_checkpoint`. The mindbot loader's own `_patch_transformers_qwen35_gguf()` captured the already-broken function as `orig_load`, and subsequent bad patchers (tvall43, etc., alphabetically after 'm') overwrote the binding sites again with incompatible versions.

**Bug 2 (loader)**: `Qwen3_5DecoderLayer` only has `self_attn` on `full_attention` layers; the 24 `linear_attention` layers use `linear_attn` (GatedDeltaNet) instead. `load_shard_spec` unconditionally accessed `layer.self_attn`, raising AttributeError on linear_attention layers.

**Bug 3 (tt-mlir, Tier B)**: Qwen3.5 is a hybrid SSM model with 24 GatedDeltaNet layers that use `torch_chunk_gated_delta_rule`. This function contains Python scan loops that unroll into an enormous StableHLO scatter graph. The TT MLIR compiler cannot handle this graph size and hangs indefinitely (previously observed at 56+ minutes). After both loader fixes were applied, the test made it to device compilation but hung in the GatedDeltaNet kernel compilation phase.

## Fix
**Loader fixes (applied, committed to remediation branch):**

1. `mindbot_ultra_4b_16b_i1_gguf/causal_lm/pytorch/loader.py`:
   - Added `_find_real_load_gguf_checkpoint()` that walks patcher closure chains to find the real `load_gguf_checkpoint` (the one with `model_to_load` parameter)
   - Added `_make_qwen35_load_gguf_checkpoint(base_load_fn)` that wraps the real function with qwen35→qwen3_5_text remapping
   - Added `_qwen35_gguf_context()` context manager that temporarily installs the correct patcher at all binding sites during `from_pretrained`
   - Updated `_patch_transformers_qwen35_gguf()` to use the real function via closure walking
   - Added `hasattr(layer, 'self_attn')` guard in `load_shard_spec` for linear_attention layers

**Compiler-stack fix (proposed):**
The TT MLIR compiler needs a scan primitive that can represent the GatedDeltaNet recurrence compactly, rather than unrolling it into a per-token scatter graph. This is new infrastructure work.

## Tier B justification
**Indicator**: new-infrastructure

The GatedDeltaNet (`torch_chunk_gated_delta_rule`) uses Python-level loops that trace into hundreds of scatter operations in StableHLO. The TT MLIR compiler has no scan primitive to represent this pattern compactly; it attempts to compile the full unrolled graph and hangs. A proper fix requires implementing a scan primitive in the compiler (new infrastructure), not a scoped pattern guard.

## Verification
- pytest exit: TIMEOUT (killed at 18 min; 12 min of compilation with no output)
- Hardware:    blackhole-p150b
- Duration:    18m (killed; compilation hung)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/mindbot_ultra_4b_16b_i1_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4a8e702ae7b7486f065f633333c670685f6dfe17 |
