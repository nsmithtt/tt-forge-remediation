# Remediation Summary: deepseek-deepseek_v3-pytorch-Bzantium_Tiny-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3/pytorch-Bzantium_Tiny-single_device-inference]

## Result
SILICON_PASS — four bugs fixed: three loader bugs (DynamicCache API, moe_infer numpy/dynamo collision, dtype mismatch) and one Tier A tt-metal CB overflow bug

## Stack layer
loader, tt-metal

## Tier
A

## Bug fingerprint
nlp-concat-heads-cb-double-buffer-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None
```

Expanded: four sequential failures were encountered and fixed:
1. `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`
2. `TorchRuntimeError: Dynamo failed to run FX node with fake tensors ... got AttributeError("'ndarray' object has no attribute 'add'")`
3. `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16`
4. `TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 2208256 B which is beyond max L1 size of 1572864 B`

## Root cause

**Loader bugs (1–3):** The `bzantium/tiny-deepseek-v3` model uses a custom `modeling_deepseek.py` loaded via `trust_remote_code=True`. This file was written for transformers 4.x:
1. `DynamicCache.get_usable_length()` was removed in transformers 5.x; the custom file still calls it at lines 798, 929, 1428.
2. `DeepseekV3MoE.moe_infer()` calls `.cpu().numpy()` on `tokens_per_expert` to drive a for-loop. During dynamo tracing with FakeTensors, numpy scalars returned from `.numpy()` collide with the globally-active `TorchFunctionOverride` in `tt_torch/torch_overrides.py`, which intercepts all torch function calls. The subsequent `start_idx + num_tokens` (where both are numpy scalars) calls `func(*args)` which returns ndarray instead of a tensor, causing dynamo to fail with "ndarray has no attribute 'add'".
3. After patching moe_infer to a static per-expert matmul, `expert_weight` inherited `topk_weight`'s float32 dtype (from the softmax gate), but `expert_out` was bfloat16; the multiply produced a float32 result that propagated as hidden_states, causing a dtype mismatch in the next layer's `q_a_proj`.

**tt-metal bug (4):** `NLPConcatHeadsProgramFactory::create()` unconditionally doubles `cb_src0_num_tiles` for the non-sharded path to enable double-buffering. For DeepSeek-V3's MLA configuration (`num_attention_heads=128`, `v_head_dim=128`): `per_tensor_tiles = 128 * 128 / 32 = 512` tiles × 2 × 2048 bytes/tile = 2,097,152 bytes > 1,572,864 bytes (Wormhole L1 limit), causing the TT_THROW.

## Fix

**Loader fixes** (`tt-xla/third_party/tt_forge_models`, branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-22`):

File: `deepseek/deepseek_v3/pytorch/loader.py`

1. Added monkey-patch at import time to restore `DynamicCache.get_usable_length` as an alias to `get_seq_length(layer_idx)`.
2. Added `_patch_moe_layers(model)` function that replaces `DeepseekV3MoE.moe_infer` with a static version: iterates `range(self.experts_per_rank)` (Python constant, dynamo unrolls), runs each expert on all tokens, gates output by `(topk_ids == i) * topk_weight` — no `.numpy()` or `.tolist()`, no device-to-host transfer.
3. In the patched `moe_infer`, added `.to(x.dtype)` on `expert_weight` after computing the routing weight sum to keep it in the model's dtype.

**tt-metal fix** (`tt-metal`, branch `remediation/deepseek-deepseek_v3-pytorch-Bzantium_Tiny-single_device-inference`):

File: `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp`

Changed unconditional `cb_src0_num_tiles *= 2` to a conditional guarded by `a.device()->l1_size_per_core()`: only double-buffer when `per_tensor_tiles * 2 * single_tile_size <= l1_size`.

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 126.90s (0:02:06)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/deepseek/deepseek_v3/pytorch/loader.py`
- `tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 888b57134fda24c0a467d3f50d9fd9f943a3dcbe |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b1ce1ab96d960473d08965864d8feb831221bbb0 |
| tt-forge-models | 61f4ce30d19863f3dc2e80c162860900a527866c |
