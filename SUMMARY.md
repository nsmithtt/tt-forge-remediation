# Remediation Summary: jetmoe-causal_lm-pytorch-8b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jetmoe/causal_lm/pytorch-8B-single_device-inference]

## Result
FAIL — PCC=0.9867 < required 0.99; ttmlir-bf16-matmul-precision-floor after loader fixes

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: Test exceeded configured timeout and was killed

Local reproduction failure: torch._dynamo.symbolic_convert.SpeculationLogDivergence —
SpeculationLog diverged at index 682 (log had 13859 entries):
- Expected: minicpmv_2_6/pytorch/loader.py:46 (CALL at ip=19)
- Actual: minicpm_o_2_6/pytorch/loader.py:48 (CALL at ip=19)

After loader fixes, remaining failure:
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.986665047464179. Required: pcc=0.99.

## Root cause
Two loader bugs combined to cause the CI timeout.

**Bug 1 — MiniCPM session contamination (loader):**
Both `minicpm_o_2_6` and `minicpmv_2_6` install `nn.Module.__getattr__ = patched_getattr`
at module-import time and never restore it. When the test framework imports all model
loaders for test discovery, this global patch persists throughout the session. During
JetMoE's Dynamo tracing, `ModuleList.__getitem__` with a slice creates a new `ModuleList`
via `add_module` → `hasattr` → `Module.__getattr__`. The chained patches from both
MiniCPM loaders cause the SpeculationLog to record different functions on first trace vs
restart, producing the SpeculationLogDivergence. Under CI this causes repeated Dynamo
restarts that run until timeout.

**Bug 2 — JetMoE expert_size.tolist() D2H hang (loader):**
`JetMoeTopKGating.forward` calls `expert_size.tolist()` causing a D2H tensor transfer
mid-forward. The comment in modeling_jetmoe.py itself documents this breaks torch.compile.
`JetMoeParallelExperts.forward` then uses `inputs.split(expert_size, dim=0)` creating
dynamically-shaped tensors, and iterates per expert with a for-loop. Both the FFN MoE
(`JetMoeMoE`) and the attention MoA (`JetMoeMoA`) use this pattern (24 layers × 2 blocks
= 48 dynamic D2H transfers per forward pass), causing XLA to either hang or trigger
excessive recompilation until CI timeout.

**Remaining precision issue (tt-mlir, Tier B):**
After fixing both loader bugs, the test runs to completion (494s) but achieves
PCC=0.9867, below the required 0.99 threshold. CPU BF16 vs FP32 gives PCC=1.000 for
4 layers, and my static MoE is bit-for-bit identical to the original on CPU (max diff=0).
The 0.013 PCC gap is caused by TT silicon's BF16 matmul precision being less accurate than
CPU. This is the known `ttmlir-bf16-matmul-precision-floor` compiler bug (tracked in
tt-xla #2861), same as observed in Gemma 7B, Qwen3 4B, GPT-J 6B, and BlackSheep 12B.

## Fix
**Fix 1 (minicpm_o_2_6, loader):** Removed module-level `nn.Module.__getattr__ =
patched_getattr`. Replaced with a `_resampler_init_weights_patch()` context manager that
applies and restores the patch exclusively during `AutoModel.from_pretrained()` inside
`load_model()`.
Files: `minicpm_o_2_6/pytorch/loader.py`

**Fix 2 (minicpmv_2_6, loader):** Same fix as minicpm_o_2_6.
Files: `minicpmv_2_6/pytorch/loader.py`

**Fix 3 (jetmoe, loader):** Added `_patched_jetmoe_moe_forward`, `_patched_jetmoa_map`,
and `_patched_jetmoa_reduce` that replace the `expert_size.tolist() + split` dynamic
pattern with a static per-expert loop: for each expert e, compute the linear on all N
tokens and weight by `gate_matrix[:, e]` (0 for non-assigned tokens). This avoids any D2H
transfer and produces all-static shapes. Applied via `_patch_jetmoe_moe()` after
`AutoModelForCausalLM.from_pretrained()`.
Files: `jetmoe/causal_lm/pytorch/loader.py`

All fixes are in branch `remediation/jetmoe-causal_lm-pytorch-8b-single_device-inference`
of the tt-forge-models repo (commit 42da03a80f).

**Proposed fix for remaining Tier B (tt-mlir):** Preserve FP32 accumulation through
StableHLO→TTIR lowering passes, or implement a higher-fidelity BF16 matmul path on
Wormhole/Blackhole. This is the same fix needed for Gemma 7B (tt-xla #2861).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting: fixing the BF16 precision floor requires preserving F32 precision
through every matmul lowering pass in the compiler stack, touching multiple files
across tt-mlir and tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    494.69s (0:08:14)
- Tier A attempts: N/A

## Files changed
- `minicpm_o_2_6/pytorch/loader.py` (MiniCPM-o session contamination fix)
- `minicpmv_2_6/pytorch/loader.py` (MiniCPM-V session contamination fix)
- `jetmoe/causal_lm/pytorch/loader.py` (JetMoE static MoE forward fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 42da03a80f |
