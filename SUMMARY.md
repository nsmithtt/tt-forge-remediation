# Remediation Summary: internlm3-causal_lm-pytorch-8B_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[internlm3/causal_lm/pytorch-8B_Instruct-single_device-inference]

## Result
FAIL — PCC=0.272 on TT silicon; root cause is ttmlir-bf16-matmul-precision-floor across 48 layers (Tier B)

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
E   ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/ttuser/hf-bringup/tt-xla/.local_venv/lib/python3.12/site-packages/transformers/utils/__init__.py)

## Root cause
**Loader bugs (two, both fixed):**

1. `LossKwargs` was never shipped in any public transformers release; the InternLM3 remote model code (written against a dev build of transformers) imports it from `transformers.utils`. Transformers 5.7.0 does not provide it.

2. `DynamicCache.to_legacy_cache()` was removed in transformers 5.x. The InternLM3 forward method calls it when `use_cache=True` (default) and `past_key_values` starts as `None` (which sets `return_legacy_cache=True`).

**Residual compiler-stack bug (Tier B):**

After both loader fixes, the model compiles and runs on TT silicon but produces PCC=0.272 (required 0.99). InternLM3-8B has 48 hidden layers with `hidden_size=4096` and `intermediate_size=10240`. The WH BF16 matmul precision floor (`ttmlir-bf16-matmul-precision-floor`) causes error that compounds over all 48 layers, resulting in dramatically lower PCC than observed in shallower models (Qwen3 4B/36 layers: PCC=0.864, Gemma 7B/28 layers: PCC=0.915). CPU BF16 reference with proper attention masking produces clean finite values; the divergence is entirely on the TT side.

## Fix
Two loader fixes in `tt_forge_models/internlm3/causal_lm/pytorch/loader.py` on branch `remediation/internlm3-causal_lm-pytorch-8B_Instruct-single_device-inference`:

1. **LossKwargs shim**: Inject a compatible `TypedDict` into `transformers.utils` at module-load time before `from_pretrained` triggers the remote model code import:
   ```python
   import transformers.utils as _tf_utils
   if not hasattr(_tf_utils, "LossKwargs"):
       class _LossKwargs(TypedDict, total=False):
           num_items_in_batch: Optional[int]
       _tf_utils.LossKwargs = _LossKwargs
   ```

2. **use_cache=False + eager attention**: Set `model.config.use_cache = False` after loading (removes the `to_legacy_cache()` path) and pass `attn_implementation="eager"` to `from_pretrained` to avoid SDPA composite-mask issues on TT.

**Proposed fix for the Tier B precision bug:**
The fix lives in `tt-mlir`: the `ttmlir-bf16-matmul-precision-floor` issue requires preserving FP32 accumulation through TTNN matmul lowering passes. This is a cross-cutting change across all matmul lowering patterns and is infeasible to backfill for large (4B+) models without hardware-level FP32 matmul support.

## Tier B justification
`cross-cutting` — fixing the BF16 matmul precision floor requires changing accumulation precision in all TTNN matmul lowering patterns throughout `tt-mlir`/`tt-metal`. This is the same issue tracked as `ttmlir-bf16-matmul-precision-floor` in prior reports (Qwen3, Gemma, GPT-J). The memory note confirms: "F32 workaround infeasible at 4B+ params" because it would require FP32 weights throughout all matmul ops.

## Verification
- pytest exit: FAIL (PCC=0.272, required 0.99)
- Hardware:    n150
- Duration:    128.09s (2:08)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/internlm3/causal_lm/pytorch/loader.py` — LossKwargs shim, use_cache=False, attn_implementation=eager

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 71bb0ea578174b1b98ce3d71c8365e0edfdd73b4 |
| tt-forge-models | 5129c7cc0dcd72df2ad20f83d095bdcdb832f1b0 |
