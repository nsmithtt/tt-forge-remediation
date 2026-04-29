# Remediation Summary: bjivanovich_qwen3_5_4b_vision_gguf-causal_lm-pytorch-4B_Vision_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bjivanovich_qwen3_5_4b_vision_gguf/causal_lm/pytorch-4B_Vision_GGUF-single_device-inference]

## Result
FAIL — PCC 0.6206 (required 0.99) after loader and evaluator fixes; SSM hybrid model numerical accuracy failure on TT silicon

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
qwen3-5-ssm-hybrid-low-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced as (with gguf 0.18.0 installed):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, a second failure surfaced:
```
TypeError: equal(): argument 'input' (position 1) must be Tensor, not Qwen3_5DynamicCache
```

After loader fix + evaluator fix:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.6206226170499918. Required: pcc=0.99.
```

## Root cause

**Loader (fixed):** The `bjivanovich/Qwen3.5-4B-Vision-GGUF` file stores architecture as `qwen35` (an SSM/Mamba-attention hybrid with `full_attention_interval=4`). The loader lacked (1) `qwen35` registration in `GGUF_TO_TRANSFORMERS_MAPPING`, and (2) the `model_to_load` kwarg required by transformers 5.x in `load_gguf_checkpoint`. Existing loaders (tvall43, mradermacher, etc.) that patch `load_gguf_checkpoint` globally at import with a version missing `**kwargs` were clobbering the binding sites before this model was loaded. The fix uses a context manager that correctly handles `**kwargs`, re-registers `qwen35` config mappings (forced, not `setdefault`, to override prior `qwen35→qwen3` aliases), detects the SSM hybrid via `qwen35.full_attention_interval`, maps `model_type` to `qwen3_5_text` (routing to `Qwen3_5ForCausalLM`), and applies a custom `TensorProcessor` for SSM weight reshaping and A_log transforms.

**Evaluator (fixed):** `Qwen3_5DynamicCache` (the hybrid KV+SSM cache for this model) does not inherit from `transformers.Cache`, so `_match_data_types` in `torch_comparison_evaluator.py` did not invoke `_cache_to_legacy` on it. `tree_map(_equal_leaf, ...)` then tried `torch.equal(Qwen3_5DynamicCache, ...)` and raised TypeError. Fix: duck-type detection (`key_cache`/`value_cache` attributes) + `_cache_to_legacy` extension to flatten per-layer KV tensors and SSM states.

**PCC failure (unfixed):** After both fixes, the model compiles and runs on TT silicon (2 inference samples, ~49 s/sample on blackhole-p150b) but PCC is 0.62 (required 0.99). Root cause is unclear: the Qwen3.5 SSM hybrid uses Mamba-style linear attention layers whose scan operations and/or SSM state computations (conv1d, recurrent scan) may produce incorrect values on TT silicon, or the SSM cache tensors (conv_states, recurrent_states) have numerically large divergence while logits are actually correct. Distinguishing these requires per-tensor PCC instrumentation, which is beyond one Tier A attempt.

## Fix
**tt_forge_models** (`remediation/bjivanovich_qwen3_5_4b_vision_gguf-causal_lm-pytorch-4B_Vision_GGUF-single_device-inference`):
- `ded5b43664`: Fix `_patched_load_gguf_checkpoint` in 26 loaders to forward `**kwargs` for transformers 5.x `model_to_load` compatibility
- `2677e42da8`: Add `_qwen35_vision_gguf_context()` context manager to bjivanovich loader: SSM hybrid detection, qwen35→qwen3_5_text remapping, custom TensorProcessor for conv1d reshape and A_log transform
- `fab3311bf4`: Fix `load_shard_spec` to skip `self_attn` on `linear_attention` layers

**tt-xla** (`remediation/bjivanovich_qwen3_5_4b_vision_gguf-causal_lm-pytorch-4B_Vision_GGUF-single_device-inference`):
- `1113278ec`: Extend `_cache_to_legacy` and `_match_data_types` in `tests/infra/evaluators/torch_comparison_evaluator.py` to handle `Qwen3_5DynamicCache` (non-`Cache`-subclass hybrid cache with `key_cache`/`value_cache` lists)

## Tier B justification
cross-cutting

SSM/Mamba linear attention scan operations in the Qwen3.5 hybrid model produce PCC 0.62 on TT silicon. Root cause is either: (a) scan op numerical divergence in tt-mlir/tt-metal lowering for Mamba-style layers, or (b) the comparison now includes SSM state tensors (conv_states, recurrent_states) that are inherently more numerically sensitive than logits. Determining which requires per-tensor PCC instrumentation across all model outputs; fixing (a) would require tt-mlir/tt-metal changes to scan op lowering across potentially many files. One Tier A attempt (evaluator fix) was made and did not produce a passing test.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    4197.68s (1:09:57)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/bjivanovich_qwen3_5_4b_vision_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py` (and 25 other qwen35 loaders)
- `tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1113278ec68e95d0176f310f9388fe7372ddb496 |
| tt-forge-models | fab3311bf4883e8ca4da827702ccb6bb097ded96 |
