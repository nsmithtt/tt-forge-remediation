# Remediation Summary: drt_7b_i1_gguf-causal_lm-pytorch-7B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[drt_7b_i1_gguf/causal_lm/pytorch-7B_i1_GGUF-single_device-inference]

## Result
FAIL — Tier B WH BF16 matmul precision (PCC=0.9900 < required 0.99; BF16 floor=0.9993); loader bugs fixed

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure (gguf not installed):
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

With gguf installed (reproduced locally):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, PCC failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9899818530467346. Required: pcc=0.99.
```

## Root cause
Two bugs in sequence:

**Bug 1 (loader):** The `drt_7b_i1_gguf` loader was missing `requirements.txt`, so `gguf>=0.10.0`
was not installed in CI, causing the ImportError. When gguf is installed, a second loader bug
surfaces: 26 loaders across tt_forge_models define `_patched_load_gguf_checkpoint(gguf_path,
return_tensors=False)` without `**kwargs`. These loaders are all imported at pytest collection
time via `setup_test_discovery`, which installs their patched function globally into the
`transformers.integrations.gguf` namespace. When `drt_7b_i1_gguf` runs, transformers 5.2.0
calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` and hits the patched version
without `**kwargs` → TypeError.

**Bug 2 (tt-mlir, Tier B):** After the loader fix, the model loads and compiles but produces
PCC=0.9900 vs the required 0.99 threshold. The BF16 floor measured on CPU (CPU BF16 vs CPU
FP32) is 0.9993, well above the threshold. The additional 0.0093 PCC gap between TT silicon
and CPU BF16 is caused by WH BF16 matmul accumulation precision — the same class of bug
affecting other 7B models (tt-xla #2861). The GGUF Q4_K_M weights are dequantized to BF16
identically for both TT and CPU; the discrepancy comes solely from TT's matmul kernel
accumulation ordering in BF16.

## Fix
**Loader fix (committed):** In `tt_forge_models` branch
`remediation/drt-7b-i1-gguf-gguf-load-checkpoint-model-to-load-kwarg` (commit c0dbc4d035):

1. Added `drt_7b_i1_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.
2. Fixed 26 loaders that defined `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
   to use `(*args, **kwargs)` and pass them through to `_orig_load_gguf_checkpoint(*args, **kwargs)`.
   Affected loaders: qwen_3_5_imatrix_gguf, mradermacher_vilm_0_8b_sft_gguf,
   mradermacher_qwen3_5_{27b,4b,9b}_* (13 variants), tvall43_qwen3_5_*, gpt_oss_swallow_*,
   dmind_3_mini_i1_gguf, daniloreddy_qwen3_5_0_8b_gguf, bartowski_coniccat_qwen3_5_27b_writer_gguf,
   unified_reward_flex_qwen35_27b_gguf, mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf.

**Compiler fix (proposed, not attempted):** The WH BF16 matmul accumulation precision issue
requires preserving F32 precision through the MLIR lowering passes (tt-mlir). This is the
same fix needed for tt-xla #2861 — a cross-cutting change through every matmul lowering
pattern, making it Tier B.

## Tier B justification
cross-cutting — fixing WH BF16 matmul precision requires coordinated changes across all
matmul lowering patterns in tt-mlir to preserve F32 accumulation; this is the same
infrastructure gap identified in tt-xla #2861 (Gemma, Qwen3 BF16 floor issues).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    384.58s (0:06:24)
- Tier A attempts: N/A

## Files changed
In tt_forge_models (`remediation/drt-7b-i1-gguf-gguf-load-checkpoint-model-to-load-kwarg`):
- `drt_7b_i1_gguf/causal_lm/pytorch/requirements.txt` (new file)
- 26 loader files: `_patched_load_gguf_checkpoint` signature fix

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c0dbc4d035b3014870fb3320e5560f75cf47d0c3 |
