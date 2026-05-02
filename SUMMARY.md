# Remediation Summary: llada-pytorch-8B_Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llada/pytorch-8B_Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
llada-transformers5x-attention-mask-dynamo-graph-break

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(The real failure was buried under this harmless SWIG deprecation warning. The
actual errors encountered during debugging were:
1. AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
2. TypeError: LLaDAModelLM.tie_weights() got an unexpected keyword argument 'recompute_mapping'
3. AttributeError: 'LLaDAConfig' object has no attribute 'use_cache'
4. ValueError: Error code: 13 (from `0.0 in attention_mask` Dynamo graph break → TTIR mask shape mismatch))

## Root cause
Four loader-layer bugs, all caused by the remote LLaDA model code predating
transformers 5.x:

1. **all_tied_weights_keys missing**: `LLaDAModelLM.__init__` does not call
   `self.post_init()`, so `all_tied_weights_keys` is never set on the instance.
   transformers 5.x `_finalize_model_loading` accesses it unconditionally.

2. **tie_weights() kwargs**: `LLaDAModelLM.tie_weights(self)` takes no extra
   kwargs; transformers 5.x `_finalize_model_loading` calls
   `model.tie_weights(missing_keys=..., recompute_mapping=False)`.

3. **use_cache popped**: transformers 5.x `PreTrainedConfig.__init__` pops
   `use_cache` from kwargs (it was moved to `GenerationConfig`), so setting
   it in `LLaDAConfig.__init__` has no effect. `LLaDAModelLM.forward` reads
   `self.config.use_cache` and fails with AttributeError.

4. **`0.0 in attention_mask` Dynamo graph break**: `LLaDAModel.forward` uses
   `0.0 in attention_mask` (line 1251 of the remote modeling_llada.py) which
   calls `Tensor.__contains__` → `.item()`, breaking the Dynamo tracing graph.
   This caused `partition_fx_graph_for_cpu_fallback` to be invoked, which then
   failed with TTIR's constraint: "Attention mask at dim 2 must match query
   sequence length" — the resulting (B,1,1,S) mask shape didn't satisfy the
   (B,1,S,S) requirement.

## Fix
All fixes are in `llada/pytorch/loader.py` in the `tt_forge_models` repo,
on branch `remediation/llada-pytorch-8B_Base-single_device-inference`.

**Commit 5f9d914aa3** — Fix transformers 5.x compat for LLaDA 8B Base loader:
- Patches `PreTrainedModel._finalize_model_loading` as a context manager.
  Inside the patch: sets `all_tied_weights_keys` directly (avoiding
  `post_init()` which also needs `tie_weights` to accept kwargs), and wraps
  `LLaDAModelLM.tie_weights` to accept `**kwargs` if it doesn't already.
  Both patches are restored after `from_pretrained` completes.
- Sets `model.config.use_cache = False` explicitly after loading.

**Commit cb425f1e87** — Pre-transform attention_mask to avoid Dynamo graph break:
- Wraps `LLaDAModelLM.forward` at class level to pre-transform the binary
  `attention_mask` into an additive bias before it reaches `LLaDAModel.forward`.
  The original `0.0 in attention_mask` check is never reached.

**Commit 189cd166af** — Expand attention mask to (B,1,S,S) for TTIR compat:
- Expands the pre-transformed mask from `(B,1,1,S)` to `(B,1,S,S)` using
  `expand()` to satisfy the TTIR `scaled_dot_product_attention` constraint that
  dim 2 must match the query sequence length.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    195.41s (0:03:15)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/llada/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 189cd166af37d8a650a8382e1d7580c6866103d8 |
