# Remediation Summary: mamaylm_gemma_3_12b_it-causal_lm-pytorch-MamayLM-Gemma-3-12B-IT-v1_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mamaylm_gemma_3_12b_it/causal_lm/pytorch-MamayLM-Gemma-3-12B-IT-v1.0-single_device-inference]

## Result
FAIL — BF16 precision accumulation across 48 Gemma3 layers (PCC 0.455); same root cause as gemma3 1B precision issue (tt-xla #3860)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-gemma3

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Underlying cause surfaced during reproduction:
```
TypeError: Gemma3ForConditionalGeneration.__init__() got an unexpected keyword argument 'use_cache'
```
Then after loader fixes:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.4552559789377637. Required: pcc=0.99.
```

## Root cause

Two distinct bugs found in the loader, both fixed:

**Bug 1 (loader):** In transformers 5.x, `AutoModelForCausalLM` maps `gemma3` → `Gemma3ForConditionalGeneration`, whose `__init__` only accepts `config` (not `use_cache`). The original loader passed `use_cache=False` as a `from_pretrained` kwarg, which the framework forwarded to `__init__`, triggering a `TypeError`. Fix: always load config explicitly and set `config.text_config.use_cache = False`.

**Bug 2 (loader):** `load_shard_spec` accessed `model.model.layers`, but `Gemma3ForConditionalGeneration.model` is `Gemma3Model`, which nests transformer layers under `language_model` (a `Gemma3TextModel`). The correct path is `model.model.language_model.layers`.

**Residual compiler issue (Tier B):** After the loader fixes, the model compiles and runs on TT silicon but produces PCC 0.4552 vs required 0.99. The comparison is BF16 TT vs BF16 CPU. The same model family's 1B variant (18 layers) already has a known precision issue at PCC 0.955 (tt-xla #3860, `assert_pcc: false`). The 12B variant has 48 layers and accumulates the same per-layer BF16 precision error much more severely. The root cause is TT's BF16 matmul accumulation diverging from CPU BF16 accumulation, and fixing it requires cross-cutting changes to how BF16 matmul precision is handled in MLIR lowering for large models.

## Fix

**Loader fixes committed to tt_forge_models branch `remediation/mamaylm_gemma_3_12b_it-causal_lm-pytorch-MamayLM-Gemma-3-12B-IT-v1_0-single_device-inference`:**

1. `mamaylm_gemma_3_12b_it/causal_lm/pytorch/loader.py`: Remove `use_cache=False` from `model_kwargs`; always call `AutoConfig.from_pretrained` and set `config.text_config.use_cache = False`; also remove spurious `torch_dtype` kwarg from tokenizer loading.

2. `mamaylm_gemma_3_12b_it/causal_lm/pytorch/loader.py`: Fix `load_shard_spec` to iterate over `model.model.language_model.layers` instead of `model.model.layers`.

**Proposed compiler fix (not attempted):** The BF16 precision issue is the same as tt-xla #3860. Fixing it would require improving BF16 matmul accumulation precision in the MLIR lowering, which is a cross-cutting change affecting all BF16 models.

## Tier B justification

Which indicator applies: **cross-cutting**

The BF16 matmul precision issue affects all Gemma3 (and likely other) BF16 models proportionally to their depth. Fixing it requires coordinated changes across the MLIR lowering passes for matmul/linear operations — not a single scoped fix. The same bug (tt-xla #3860) is already open and unfixed for the 1B variant.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    264.00s (0:04:24) for the final run
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/mamaylm_gemma_3_12b_it/causal_lm/pytorch/loader.py` (2 commits in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5d95c98b34143543f7cf5389bed953eed46b6168 |
