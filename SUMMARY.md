# Remediation Summary: deepseek_r1_distill_qwen_7b_gguf-causal_lm-pytorch-eaddario_Distill_Qwen_7B_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_7b_gguf/causal_lm/pytorch-eaddario_Distill_Qwen_7B_GGUF-single_device-inference]

## Result
FAIL — SDPA k_chunk_size bug corrupts attention for seq_len=26 < 32; loader TypeError fixed but PCC=0.9123 still fails 0.99

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
sdpa-k-chunk-size-lt-32

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9123406252652001. Required: pcc=0.95.

(Before loader fix: TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause

Two bugs were identified in sequence:

**Bug 1 (loader — fixed):** 26 GGUF model loaders in `tt_forge_models` monkey-patch `load_gguf_checkpoint` at module-import time. Their patched function signatures were `(gguf_path, return_tensors=False)` — missing `model_to_load=None`. transformers 5.2.0 added `model_to_load` to the real function and calls it with that kwarg from `modeling_utils.py:4016`:
```
state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)
```
When pytest collects all tests, the broken patches are applied globally, so even loaders without the patch (like deepseek) are affected because other loaders' patches overwrite the function pointer. Fixed in `tt_forge_models` commit `f6946e1294`.

**Bug 2 (tt-metal — unfixed):** After the loader fix, the test runs but produces PCC=0.9123 < 0.99. The root cause is the SDPA k_chunk_size constraint. The tokenized input has seq_len=26 tokens (< k_chunk_size=32). In `sdpa_program_factory.cpp`, `padded_Sk = ceil(26/32)*32 = 32`. The user provides an explicit attention mask of shape [1, 1, 26, 26].

In `reader_interleaved.cpp`, the tile-level validity check is:
```cpp
const bool k_valid = !use_padded_mask || (global_k_tile < valid_Skt);
```
For Sk=26, `valid_Skt=ceil(26/32)=1`, `Skt=1`, so `global_k_tile=0 < valid_Skt=1` — the tile is marked valid and the actual mask tile is read. However, the user-provided mask is [26, 26] logical; when tiled to [32, 32] physical, columns 26..31 are zero-padded (mask=0, i.e. no masking). The SDPA softmax then assigns exp(0)=1 to 6 padded K positions per query, diluting valid attention weights and corrupting the output.

The same bug was reported for the non-GGUF variant (`bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF`) in a prior report (bug fingerprint: `sdpa-k-chunk-size-lt-32`), where even switching to eager SDPA yielded PCC=0.970, still failing 0.99. The GGUF model produces 0.9123 which is even further from threshold.

## Fix

**Loader fix (committed):** Modified all 26 GGUF loaders in `tt_forge_models` that define `_patched_load_gguf_checkpoint` to add `model_to_load=None` to the function signature and pass it through to `_orig_load_gguf_checkpoint`. Files modified (representative):
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- (25 additional loader files with the same pattern)

Committed as `f6946e1294` on branch `remediation/deepseek_r1_distill_qwen_7b_gguf-eaddario-single_device-inference` in `tt_forge_models`, with the tt-xla submodule pointer updated at `9f37f6ab5`.

**SDPA fix (not attempted — Tier B):** The fix would require modifying the SDPA reader kernel in `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp` to detect sub-tile padding (when `Sk % TILE_HEIGHT != 0` and a user-provided mask is given) and apply -inf masking to the padded K columns within the first/last tile. This is a cross-cutting kernel change: it would also require updates to `sdpa_program_factory.cpp` (to pass the sub-tile mask boundary info) and potentially `dataflow_common.hpp`, touching 2-3 files with kernel-specific tile layout knowledge. An existing report filed the same bug as FAIL.

## Tier B justification

Indicator: **cross-cutting** — The fix requires coordinating changes across the SDPA program factory (to compute and pass `valid_Sk_within_tile`), the reader kernel (to conditionally fill -inf into sub-tile padded K columns), and potentially the mask generation path in `dataflow_common.hpp`. Additionally, the prior non-GGUF report for the same model with eager SDPA still yielded PCC=0.970 (failing 0.99), raising doubt about whether fixing SDPA alone would be sufficient.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    not-run (test not re-run after loader fix due to SDPA bug classification)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/` — 26 GGUF loader files: added `model_to_load=None` parameter to `_patched_load_gguf_checkpoint` and pass-through to `_orig_load_gguf_checkpoint`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9f37f6ab5f14f730396ae5be4a46693aab8ad789 |
| tt-forge-models | f6946e1294d67b88affda1f6a20ca9829dbabc31 |
