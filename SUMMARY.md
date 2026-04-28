# Remediation Summary: deepseek/janus/pytorch-Janus_1_3B-single_device-inference

## Skill version
12

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/janus/pytorch-Janus_1_3B-single_device-inference]

## Result
FAIL — `index_put` with boolean mask fails at TT XLA runtime: "Failed to get computation by hash"

## Failure
The image processor of type `VLMImageProcessor` is now loaded as a fast processor by
default, even if the model checkpoint was saved with a slow processor. This is a
breaking change and may produce slightly different outputs. To continue using the slow
processor, instantiate this class with `use_fast=False`.

## Root cause

Two layers of bugs were found and fixed; a third is the compiler-stack bug that
blocks silicon pass.

**Loader-layer bugs (fixed in tt_forge_models):**

1. **Missing `janus` package dependency** — `deepseek-ai/Janus-1.3B` uses
   `model_type: multi_modality` but ships no Python files on HuggingFace, so
   `trust_remote_code=True` cannot resolve the class. The janus package from
   DeepSeek's GitHub registers `MultiModalityCausalLM` and `MultiModalityConfig`
   with AutoModelForCausalLM / AutoConfig.

2. **janus siglip_vit + transformers 5.x meta-device init** — `siglip_vit.py:391`
   calls `torch.linspace(...).item()` during `VisionTransformer.__init__`. Transformers
   5.x always initialises models inside `torch.device("meta")`, making every tensor a
   meta tensor on which `.item()` is forbidden. Fixed by redirecting
   `janus.models.siglip_vit.torch.linspace` to force `device="cpu"`.

3. **flash_attention_2 baked into language_config** — the model config embeds
   `"_attn_implementation": "flash_attention_2"`. No flash_attn package is available.
   Fixed by overriding `config.language_config._attn_implementation = "eager"` before
   loading.

4. **Missing `all_tied_weights_keys` on MultiModalityCausalLM** — transformers 5.x
   `_finalize_model_loading` calls `_adjust_tied_keys_with_tied_pointers` which
   accesses `self.all_tied_weights_keys`. Because `MultiModalityConfig` does not set
   `tie_word_embeddings`, `get_expanded_tied_weights_keys` returns `{}` but the
   assignment is silently skipped, leaving the attribute unset. Fixed by guarding the
   attribute in a monkey-patched `_adjust_tied_keys_with_tied_pointers`.

5. **VLMImageProcessor fast-processor FutureWarning** — `VLChatProcessor.from_pretrained`
   triggers a transformers 5.x FutureWarning that was being captured as the reported
   failure. Fixed by passing `use_fast=False`.

6. **Incorrect processor + input construction** — the old loader called
   `AutoProcessor.from_pretrained` which returned only a `TokenizersBackend` (no image
   processor). Fixed by using `VLChatProcessor.from_pretrained` directly and building
   inputs in the janus conversation format expected by `prepare_inputs_embeds`.

**Compiler-layer bug (unfixed, root cause of FAIL):**

`prepare_inputs_embeds` (janus `modeling_vlm.py:258`) performs a boolean masked
scatter:

```python
inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]
```

This lowers to `index_put` in torch.compile / dynamo. The TT XLA dynamo bridge
compiles this subgraph and assigns it hash `d5f9e10409c8736332ba54c6bafa9735`, but the
compiled graph is never successfully stored in the XLA LRU cache. When
`torch_xla._XLAC._run_cached_graph` tries to look up the graph by hash, it fails
deterministically with:

```
RuntimeError: Check failed: cachedComputation: Failed to get computation by hash
d5f9e10409c8736332ba54c6bafa9735. Maybe the entry get kicked out of the LRU cache
```

The same hash reproduces across runs and device resets, confirming this is a
compilation failure rather than a cache eviction.

**Layer:** tt-xla (compiler frontend / dynamo bridge).

**Hypothesis for fix:** The `stablehlo` representation of `index_put` with a boolean
index tensor (scatter with non-static indices) is likely not supported by the tt-mlir
lowering path. The compiled graph silently fails to materialise, leaving the cache
entry absent. The fix should be in tt-mlir to support `stablehlo.scatter` with dynamic
boolean index tensors, or in the tt-xla dynamo bridge to surface compilation errors
rather than silently dropping the graph.

## Fix
Loader fixes applied in `tt-xla/third_party/tt_forge_models` on branch
`remediation/deepseek-janus-pytorch-Janus_1_3B-single_device-inference`:

- `deepseek/janus/pytorch/requirements.txt` — added `git+https://github.com/deepseek-ai/Janus.git`
- `deepseek/janus/pytorch/loader.py` — rewrote with:
  - `janus.models` import for architecture registration
  - `siglip_vit.torch.linspace` monkey-patch (CPU device override)
  - `config.language_config._attn_implementation = "eager"` before load
  - `_adjust_tied_keys_with_tied_pointers` guard for missing attribute
  - `VLChatProcessor.from_pretrained(use_fast=False)` for processor
  - `JanusForwardWrapper` that calls `prepare_inputs_embeds` then `language_model`
  - Updated `load_inputs` to build janus conversation-format inputs

These are not forbidden workarounds; they are genuine transformers 5.x
compatibility fixes and a missing dependency.

The compiler-stack bug (`index_put` / `stablehlo.scatter` LRU cache failure)
is left unfixed. A model-level workaround (e.g. replacing the masked scatter with
`torch.where`) would constitute changing the model's computation to dodge a compiler
limitation, which is a forbidden workaround.

## Verification
```
FAILED tests/runner/test_models.py::test_all_models_torch[deepseek/janus/pytorch-Janus_1_3B-single_device-inference]
RuntimeError: Check failed: cachedComputation: Failed to get computation by hash
d5f9e10409c8736332ba54c6bafa9735. Maybe the entry get kicked out of the LRU cache
```
Wall clock: ~170 s (2 runs, consistent). Hardware: n150.

## Files changed
- `third_party/tt_forge_models/deepseek/janus/pytorch/loader.py`
- `third_party/tt_forge_models/deepseek/janus/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 859809f61548b25befb14c1fd784396f7c0fabc1 |
