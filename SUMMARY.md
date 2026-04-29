# Remediation Summary: bd3lm/masked_lm/pytorch-owt-block_size4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bd3lm/masked_lm/pytorch-owt-block_size4-single_device-inference]

## Result
SILICON_PASS — four loader-layer compatibility fixes for transformers 5.x and torch.compile

## Stack layer
loader

## Tier
A

## Bug fingerprint
bd3lm-transformers5-post-init-missing-modulate-dynamo-rotary-cache-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: 'BD3LM' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?
```
(original failure; four subsequent loader issues also fixed in the same pass)

## Root cause
Four distinct bugs, all in the BD3LM remote-code loader layer:

1. **post_init() not called** — Transformers 5.x requires every
   `PreTrainedModel` subclass to call `self.post_init()` at the end of
   `__init__` so that `all_tied_weights_keys` (and related fields) are set
   before `_finalize_model_loading` runs.  The BD3LM remote-code `__init__`
   was written for an older transformers API and omits this call, causing
   `AttributeError` in `_adjust_tied_keys_with_tied_pointers`.

2. **Meta-device attention mask** — Transformers 5.x initialises models on
   a meta device during `from_pretrained`'s lazy-load phase.
   `DITBackbone.gen_mask()` stores `self.mask` as a plain Python attribute
   (not a `register_buffer`), so it is never materialised from meta.
   Calling `.to(x.device)` on a meta tensor raises
   `NotImplementedError: Cannot copy out of meta tensor`.

3. **modulate_fused JIT/dynamo name shadowing** — `modeling_bd3lm.py`
   defines `modulate()` twice; the `@torch.jit.script`-decorated
   `modulate_fused()` compiles the correct first definition,
   but `torch.compile`/dynamo resolves the name at Python trace time and
   picks up the shadowing second definition (which adds `.unsqueeze(1)`),
   producing a 4-D tensor that breaks `einops.rearrange` inside `get_qkv`.

4. **Rotary cache device mismatch** — `Rotary.forward()` caches cos/sin
   keyed only on sequence length; after the CPU reference run the cache
   holds CPU tensors.  When torch.compile traces for the XLA device with the
   same seq_len, the stale CPU cache is returned, causing
   "found two different devices xla:0, cpu" in FakeTensor propagation.

Additional issues fixed: GPT-2 tokenizer has no pad_token (set to eos_token);
TimestepEmbedder hard-codes float32, making bfloat16 loading incompatible.

## Fix
All fixes in `tt_forge_models/bd3lm/masked_lm/pytorch/loader.py`:

- Patch 1: wrap BD3LM.__init__ to call self.post_init() when all_tied_weights_keys is absent
- Patch 2: regenerate meta-device backbone.mask after from_pretrained
- Patch 3: replace @torch.jit.script modulate_fused with plain Python function
- Patch 4: patch Rotary.forward() to invalidate cache on device change
- Set pad_token = eos_token in tokenizer
- Do not forward bfloat16 dtype_override; always create timesteps in float32

Repos changed:
- tt-forge-models branch remediation/bd3lm-masked_lm-pytorch-owt-block_size4-single_device-inference
- tt-xla branch remediation/bd3lm-masked_lm-pytorch-owt-block_size4-single_device-inference (submodule pointer bump)

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    76.94s
- Tier A attempts: 1

## Files changed
- tt-forge-models/bd3lm/masked_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2eae6c6e42311711bbeae3fe3b5204f06240ecad |
| tt-forge-models | 5f9251b37d29e220ec41e630600ab5d7c954748c |
