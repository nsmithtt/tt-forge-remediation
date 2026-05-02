# Remediation Summary: moody_real_mix_v4_dpo_gguf-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[moody_real_mix_v4_dpo_gguf/pytorch-Q4_K_M-single_device-inference]

## Result
FAIL — TT device execution aborts under 30 s watchdog after MLIR compilation; hang mechanism unknown (Tier B: tt-metal device hang, unknown mechanism)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
ttmetal-device-execution-hang-lumina2-gguf

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `raise ImportError(Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.)`
(also manifested as `OSError: Gthalmie1/moody-real-mix-v4-dpo-gguf does not appear to have a file named resolve/main/moodyRealMix_zitV4DPO_q4_k_m.gguf.` depending on gguf install state)

Terminal (after all loader fixes): `Fatal Python error: Aborted` during `FlatbufferLoadedExecutableInstance::Execute`
(`_call_experimental_compile` in `tt_torch/backend/backend.py`). With
`TT_METAL_OPERATION_TIMEOUT_SECONDS=30`, the process aborts silently after ~30 s; the
abort propagates through a `noexcept` boundary in `invoke_noexcept` → `std::terminate()`.

## Root cause
**Loader layer (fixed):** Five loader bugs fixed in `moody_real_mix_v4_dpo_gguf/pytorch/loader.py`
(carried from previous remediation branch, unchanged):
1. Missing `requirements.txt` — `gguf>=0.10.0` not declared.
2. Broken URL: `resolve/main/...gguf` → `hf_hub_download(repo_id, filename)`.
3. `GGUFParameter.as_tensor` Dynamo infinite recursion — fixed with `DisableTorchFunctionSubclass` guard.
4. Lumina2 MHA vs GQA: `split_with_sizes` dimension mismatch — patched QKV split.
5. Architecture kwargs missing from `from_single_file` — shape mismatch `[2304]` vs `[3840]`.
6. Complex `<f64>` RoPE causing `Error code: 13` — replaced with real-arithmetic RoPE (cos/sin float32 tuples).
7. `aten.view.dtype` in Q4_K dequantizer causing `Error code: 13` — `_dequantize_gguf_and_restore_linear` + `Module.to(compute_dtype)`.
8. `for-loop` / `.tolist()` patterns in rope_embedder and transformer forward causing segfault — replaced with tensor-based cat/slice.

**Terminal (unfixed, Tier B):** After all loader fixes, the model compiles via MLIR then the TT device
aborts under the 30 s watchdog. The process exits with `Fatal Python error: Aborted` from `std::terminate`
in `invoke_noexcept`. No TT_FATAL or diagnostic message is emitted before the abort. The Lumina2
Transformer2DModel (hidden_size=3840, 30 layers) is a large diffusion transformer; whether the hang is
in MLIR compilation, CB allocation, dispatch, or device queue starvation is unknown without runtime
instrumentation. The result is identical on the current compiler branch
(`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-39`) as on the previous report.

## Fix
**Loader fixes (committed, 5 commits on remediation branch in tt_forge_models):**
- `moody_real_mix_v4_dpo_gguf/pytorch/requirements.txt` — created with `gguf>=0.10.0`
- `moody_real_mix_v4_dpo_gguf/pytorch/loader.py` — all 8 loader bugs fixed

**Terminal compiler-stack fix (proposed, not attempted):**
Investigate why the TTNN binary for Lumina2Transformer2DModel hangs/aborts on dispatch. Possible causes:
a specific TTNN op dispatch deadlock, CB allocation hang, or device queue starvation for a 30-layer
hidden_size=3840 DiT. Needs tt-metal dispatch/runtime instrumentation.

## Tier B justification
`internal-error-unknown-mechanism` — the device aborts silently with no TT_FATAL or diagnostic output.
The mechanism (deadlock vs. infinite loop vs. dispatch queue exhaustion) is unknown and requires runtime
instrumentation to diagnose. This cannot be scoped to a single file or function without that diagnosis.

## Verification
- pytest exit: FAIL (Fatal Python error: Aborted)
- Hardware: blackhole-p150b
- Duration: ~30 s (watchdog timeout)
- Tier A attempts: N/A

## Files changed
- `moody_real_mix_v4_dpo_gguf/pytorch/requirements.txt` (created, in tt_forge_models)
- `moody_real_mix_v4_dpo_gguf/pytorch/loader.py` (loader fixes, in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e21c9bc5e1b87427a59b98731099d9ab7823414b |
