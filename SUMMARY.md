# Remediation Summary: abhiray_qwen3_5_9b_abliterated_gguf-causal_lm-pytorch-9B_Abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch-9B_Abliterated_GGUF-single_device-inference]

## Result
FAIL — pcc=0.82 after all loader fixes; ttmlir-f32-precision-not-preserved in Qwen3.5 GatedDeltaNet recurrent computation (Tier B)

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
```
FAILED tests/runner/test_models.py::test_all_models_torch[abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch-9B_Abliterated_GGUF-single_device-inference]
AssertionError: PCC check failed: pcc=0.82, required=0.99
```
(Preceded by a series of loader-layer errors fixed in the remediation branch, including TypeError from model_to_load kwarg, pcc=nan from MambaTensorProcessor NaN corruption, and near-zero PCC from Qwen3_5DynamicCache in outputs.)

## Root cause
Multiple loader bugs were found and fixed (see Fix section below). After all loader fixes, pcc=0.82 remains. The residual error is in tt-mlir: TT hardware BF16 matmuls accumulate in BF16, while the CPU reference accumulates in FP32. The Qwen3.5-9B hybrid model has 28 GatedDeltaNet (SSM) layers out of 36 total. Each GatedDeltaNet forward pass performs multiple linear projections and chunk-wise recurrent state updates; BF16 rounding error accumulates across 4 sequence chunks × 28 layers, compounding to a pcc gap of ~0.17. This is the ttmlir-f32-precision-not-preserved pattern: the compiler does not preserve FP32 accumulation precision through the lowering pipeline.

## Fix
All loader-layer bugs were fixed in `tt-xla/third_party/tt_forge_models` on branch `remediation/abhiray_qwen3_5_9b_abliterated_gguf-causal_lm-pytorch-9B_Abliterated_GGUF-single_device-inference` of the tt-forge-models submodule (commit `277e57404c`).

**Fix 1 — qwen35 GGUF architecture not registered**
`abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`
Added `_patch_qwen35_support()` which registers "qwen35" in `GGUF_SUPPORTED_ARCHITECTURES` and populates `GGUF_TO_TRANSFORMERS_MAPPING` from the qwen3 entry, adding `full_attention_interval` to the config section. Qwen3.5 uses GGUF `general.architecture = qwen35` (arch id 34); without this registration `load_gguf_checkpoint` raises KeyError.

**Fix 2 — model_to_load kwarg not forwarded (gguf-load-checkpoint-model-to-load-kwarg)**
`abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`
Wrapped `load_gguf_checkpoint` with `_patched_load_gguf_checkpoint(*args, **kwargs)` forwarding all arguments. Other GGUF loaders in the test collection also patch `load_gguf_checkpoint` at binding sites with narrow signatures; transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` which crashes any narrow-signature patcher.

**Fix 3 — model_type not remapped to qwen3_5_text**
`abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`
After `load_gguf_checkpoint` returns `cfg["model_type"] = "qwen35"` (or "qwen3" if another patcher ran first), remapped to `qwen3_5_text` when `full_attention_interval` is present. This selects `Qwen3_5ForCausalLM` via AutoModelForCausalLM, which implements the hybrid SSM+full-attention architecture.

**Fix 4 — MambaTensorProcessor NaN on ssm_alpha.weight**
`abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`
Added `_Qwen35TensorProcessor` replacing `MambaTensorProcessor` for the qwen35 architecture. `MambaTensorProcessor.process()` checks `"ssm_a" in name` as a substring, which also matches `blk.N.ssm_alpha.weight`. After Q4_K_M dequantization `ssm_alpha.weight` has positive values; `np.log(-positive) = NaN` corrupts those weights and produces pcc=nan. The custom processor applies `np.log(-weights)` only when `name.endswith(".ssm_a")`.

**Fix 5 — Qwen3_5DynamicCache in model output (use_cache=True)**
`abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`
Added `model.config.use_cache = False` after loading the model. With `use_cache=True`, forward returns `(logits, Qwen3_5DynamicCache)`. `Qwen3_5DynamicCache` is not a `transformers.Cache` subclass, so the test evaluator cannot convert it; PCC is taken over all outputs and the cache object pulls the score to near 0.

**Fix 6 — load_shard_spec missing hybrid layer handling**
`abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`
Added handling for both `layer.self_attn` (full attention, 8 layers) and `layer.linear_attn` (GatedDeltaNet SSM, 28 layers) in `load_shard_spec`.

**Remaining bug — ttmlir-f32-precision-not-preserved (Tier B, not fixed)**
After all loader fixes, pcc=0.82 on TT silicon. The fix would require preserving FP32 accumulation through all matmul lowerings in tt-mlir, a cross-cutting change touching many files across tt-mlir and tt-metal. Not attempted.

## Tier B justification
cross-cutting: Preserving FP32 accumulation precision through the GatedDeltaNet recurrent computation requires changes to the matmul lowering and accumulation-type selection in tt-mlir, affecting multiple lowering passes and kernel configurations across tt-mlir and tt-metal. The per-op impact is small but the change surface is large.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~1740s for the failing run with all loader fixes applied
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/abhiray_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py` (all fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | (unchanged from main) |
| tt-mlir         | (unchanged from main) |
| tt-xla          | 52057b5af38b45acb1740d4b239c530354f61d67 |
| tt-forge-models | 277e57404c (on remediation branch in tt-forge-models) |
