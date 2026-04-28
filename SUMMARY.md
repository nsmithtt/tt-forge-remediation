# Remediation Summary: jina_clip_v2-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jina_clip_v2/pytorch-Base-single_device-inference]

## Result
SILICON_PASS — four loader bugs and one tt-mlir compiler bug fixed; pytest exits PASS on silicon

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
gather-to-slice-concat-maxindex-zero

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original failure was `torch._dynamo.exc.SpeculationLogDivergence: ... have changed on restart.` — Dynamo's recompile count exceeded the limit (18 > 8) because `EVAVisionTransformer.forward_features` created a new `partial(rope.forward, ...)` object on every eval call, making Dynamo's guard non-stable.

After fixing that, the next failure was `'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 7 vs. 13.` from `StableHLOGatherToSliceRepeatConcatPattern` applied to the XLM-RoBERTa `token_type_ids` embedding lookup.

## Root cause

Five bugs were found and fixed across the loader (4) and tt-mlir (1):

**Loader bug 1 — EVA rope partial accumulation (SpeculationLogDivergence)**
`EVAVisionTransformer.forward_features` called `self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)` on every eval call, creating a new callable object each time. Dynamo's guard checked object identity and saw 18 different callables (> limit 8), forcing fallback to eager with wrong results.

**Loader bug 2 — EVA `freqs_cos`/`freqs_sin` uninitialized (NaN vision embeddings)**
`VisionRotaryEmbeddingFast` registers `freqs_cos`/`freqs_sin` as `persistent=False` buffers. Under transformers 5.x meta-device init, these are absent from the checkpoint and materialized with garbage data (NaN), corrupting EVA vision encoder outputs.

**Loader bug 3 — XLM-RoBERTa `inv_freq` uninitialized (NaN text embeddings)**
`RotaryEmbedding.inv_freq` in the XLM-RoBERTa text encoder is also `persistent=False`. Same root cause as bug 2: meta-device init leaves it with garbage float32 values (magnitudes up to 1.6×10³⁰), causing `_update_cos_sin_cache` to produce NaN cosine/sine caches, which corrupt rotary embedding application in every text encoder layer.

**Loader bug 4 — LoRA adapter_mask dynamic dispatch (PCC=-1.0 / incompatible with TT static compilation)**
`mha.py` and `mlp.py` dispatch LoRA via `adapter_mask`: they call `torch.unique(adapter_mask)`, then `nonzero()` for each task, then write back results via in-place scatter (`qkv[task_indices] = task_qkv`). These dynamic tensor operations are incompatible with TT's static-shape compiler and produce pcc=-1.0 text embeddings.

**tt-mlir bug — `StableHLOGatherToSliceRepeatConcatPattern` double-counts when maxIndex == 0 (shape mismatch)**
When the gather operand has exactly one row in the indexed dimension (`sliceSizes[dim] == inputShape[dim]`), `maxIndex = inputShape[dim] - sliceSizes[dim] = 0`. Every index simultaneously satisfies `(index == 0)` AND `(index == maxIndex == 0)`, so both `starts` and `ends` count all N indices. After decrement: `starts = N-1`, `ends = N-1`. The concat assembles `(N-1) + 1 + (N-1) = 2N-1` elements, but the gather output type declares N elements. The TTIR verifier catches the mismatch. Triggered by the XLM-RoBERTa `token_type_ids` embedding table which has shape `[1, 768]`.

## Fix

**tt_forge_models — `jina_clip_v2/pytorch/loader.py`** (remediation branch in tenstorrent/tt-forge-models):

1. `_recompute_rope_buffers(model)` — recomputes `VisionRotaryEmbeddingFast.freqs_cos/freqs_sin` on CPU from config parameters after meta-device init.

2. `_recompute_xlm_roberta_rope_buffers(model)` — recomputes `RotaryEmbedding.inv_freq` for all XLM-RoBERTa rotary embedding modules on CPU by calling the module's own `_compute_inv_freq` helper.

3. `_fix_eva_rope_forward_accumulation(model)` — replaces each `EVAVisionTransformer.forward_features` with a closure that uses a single pre-created stable `partial(rope.forward, patch_indices_keep=None)` object, preventing Dynamo guard churn.

4. `_merge_lora_for_default_task(model)` — for inference with the default task (task_id=0), pre-merges each LoRA delta (`lora_B @ lora_A * scaling`) directly into the base weight using `torch.nn.utils.parametrize.remove_parametrizations`, then sets `text_model._default_loraid = None` so `hf_model.py` never constructs `adapter_mask`. Every LoRA module becomes a plain `nn.Linear`/`nn.Embedding` on the static code path.

**tt-mlir — `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`** (remediation branch in tenstorrent/tt-mlir):

`StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite`: added an early `notifyMatchFailure` when `maxIndex == 0`, allowing the lower-priority `StableHLOGatherToEmbeddingPattern` to handle the single-row operand case correctly.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    231.15s (0:03:51)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/jina_clip_v2/pytorch/loader.py`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0b70ff4a7fa0a73f34913169e3a27693feacb31c |
| tt-xla          | 451b64f064b6383e8ac56e56d7ff4ed54a1e2ac7 |
| tt-forge-models | 957b9d540df50c45305024641a4e2de210ed4a5c |
