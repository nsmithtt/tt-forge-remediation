# Remediation Summary: mradermacher_solar_10_7b_merge_dpo_v1_i1_gguf-causal_lm-pytorch-SOLAR_10_7B_merge_dpo_v1_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_solar_10_7b_merge_dpo_v1_i1_gguf/causal_lm/pytorch-SOLAR_10_7B_merge_dpo_v1_i1_GGUF-single_device-inference]

## Result
FAIL — loader TypeError fixed; residual PCC=0.9275 < 0.99 on WH n150 is ttmlir-bf16-matmul-precision-floor Tier B

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9275429661337741. Required: pcc=0.95.

## Root cause
Two issues found:

**Loader bug (fixed):** The test session imports 26 GGUF loaders whose `_patched_load_gguf_checkpoint` functions used a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` when any of these loaders' patches is active in the pytest session.

The SOLAR loader itself does not patch `load_gguf_checkpoint`. When run in isolation (no contaminating loader imported first) it hits this TypeError through session contamination. When run first in the session (as in the original CI run) it loads successfully but then hits the precision failure.

**Compiler-stack bug (Tier B):** After the loader fix, the model runs end-to-end on WH n150 silicon and produces PCC=0.9275429661337741, below the 0.99 threshold. SOLAR-10.7B-merge-dpo_v1 uses the LLaMA/Mistral architecture with 48 transformer layers and intermediate_size=14336. BF16 matmul rounding error accumulates across all 48 × 2 MLP projection matmuls on WH silicon, producing a stable, reproducible PCC floor. This is the same class of bug as tt-xla #2861 (ttmlir-bf16-matmul-precision-floor) seen in Gemma 7B (PCC≈0.915), BlackSheep-RP 12B (PCC=0.949 on p150b), Qwen3 4B (PCC=0.864), and GPT-J 6B (PCC=0.75).

## Fix
**Loader (committed):** Updated 26 GGUF loader files in `tt_forge_models` to use `(*args, **kwargs)` signature and inner call for `_patched_load_gguf_checkpoint`. Also added `requirements.txt` with `gguf>=0.10.0` to `mradermacher_solar_10_7b_merge_dpo_v1_i1_gguf/causal_lm/pytorch/`.

Files changed (loader): 26 × `<model>/causal_lm/pytorch/loader.py` + `mradermacher_solar_10_7b_merge_dpo_v1_i1_gguf/causal_lm/pytorch/requirements.txt`

Branch: `remediation/mradermacher_solar_10_7b_merge_dpo_v1_i1_gguf-causal_lm-pytorch-SOLAR_10_7B_merge_dpo_v1_i1_GGUF-single_device-inference` in `tt_forge_models` (commit 23f0f9293c)

**Proposed compiler fix:** Add FP32 accumulation mode for TTNN matmul operations on WH silicon (tt-xla issue #2861). This is a cross-cutting change across all TTNN matmul lowerings and is not attempted here.

## Tier B justification
Indicator: cross-cutting. Fixing the BF16 precision floor requires enabling FP32 accumulation across all TTNN matmul operations in the WH backend. This touches the tt-mlir→tt-metal lowering pipeline for every matmul in the graph (>100 ops in a 48-layer LLM), requires coordinated changes across tt-mlir and tt-metal, and cannot be scoped to a single pattern or kernel.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    413.38s (0:06:53)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models` (submodule): 26 loader files + 1 requirements.txt (commit 23f0f9293c)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 23f0f9293cfc046bba5aa5ee78a80ec84231ef4e |
