# Remediation Summary: babylm_baseline_gpt_bert_mixed-causal_lm-pytorch-100M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[babylm_baseline_gpt_bert_mixed/causal_lm/pytorch-100M-single_device-inference]

## Result
SILICON_PASS — all five loader bugs fixed; test passes on silicon in 50s.

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-gpt-bert-mixed-loading-and-inplacesetslice-dynamo

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
torch._dynamo.exc.BackendCompilerFailed: backend='tt' raised:
RuntimeError: contracted dimensions need to match, but first has size 2 in dim -1 and second has size 0 in dim 0

While executing %tensordot ... (args = (%self___model_dwa_modules__modules__alphas_0, %set__1, [-1], [0]), ...)
```

## Root cause
Five bugs in the loader, all caused by the model's remote code being incompatible with
transformers 5.x and torch.compile:

**Bug 1 (pre-existing, from base branch):** `ModelConfig.to_json_string()` calls
`json.dumps(self.__dict__)` which fails when `torch_dtype=torch.bfloat16` is in
model_kwargs because `torch.dtype` is not JSON-serializable.

**Bug 2:** `GPTBERTForCausalLM.__init__` never calls `self.post_init()` (required by
transformers 5.x). `post_init()` is the only place that sets `all_tied_weights_keys`;
without it, `_finalize_model_loading` raises `AttributeError`.

**Bug 3:** `GPTBERTPreTrainedModel._init_weights` unconditionally calls
`module.bias.data.zero_()` on every `nn.LayerNorm`, but many use
`elementwise_affine=False` (no bias). `_initialize_missing_keys` triggers `_init_weights`
for uninitialized modules, causing a crash on the bias-less layers.

**Bug 4:** Transformers 5.x creates models on the meta device even when
`low_cpu_mem_usage=False`. `_move_missing_keys_from_meta_to_device` overwrites all
non-persistent buffers with `torch.empty_like` (garbage). `Attention.position_indices`
is a non-persistent buffer computed in `__init__`; after loading it contains random
values that cause index-out-of-bounds in `F.embedding`.

**Bug 5:** `DWAModules` uses `InPlaceSetSlice`, a custom `torch.autograd.Function`
that calls `torch.Tensor().set_()` to create a tensor with aliased storage. When
`torch.compile`/dynamo traces this, FakeTensor cannot propagate shape through `set_()`;
the returned tensor retains its initial empty shape (size 0), so the subsequent
`torch.tensordot(alphas[block_idx], accumulator[1], dims=1)` fails because the first
dimension of the accumulator (size 0) does not match the alpha size.

## Fix
All fixes are in the loader:
`babylm_baseline_gpt_bert_mixed/causal_lm/pytorch/loader.py`
in the `tt-forge-models` repo on branch
`remediation/babylm_baseline_gpt_bert_mixed-causal_lm-pytorch-100M-single_device-inference`.

**Bug 1 (base branch):** Temporarily monkey-patch `json.JSONEncoder.default` to
serialize `torch.dtype` as a string during `from_pretrained`.

**Bug 2:** In a `_patched_finalize` wrapper around `_PTM._finalize_model_loading`,
seed `model.all_tied_weights_keys = {}` before `_orig_finalize` runs when the
attribute is missing.

**Bug 3:** In the same `_patched_finalize`, iterate all modules before calling
`_orig_finalize` and set `m._is_hf_initialized = True` on any `nn.LayerNorm` with
`elementwise_affine=False`, causing `_initialize_missing_keys` to skip them.

**Bug 4:** After `_orig_finalize` returns, recompute `position_indices` for every
`Attention` module that has both `make_log_bucket_position` and `config`, using the
same formula as the original `__init__`, and re-register the buffer as non-persistent.

**Bug 5:** `torch.ops` and `torch.classes` both expose a `DWAModules` attribute that
shadows the real class when iterating `sys.modules` (they appear earlier in iteration
order). The fix is to obtain the actual class from the loaded model instance
(`type(model.model.dwa_modules)`), then replace `DWAModules.forward` and
`DWAModules.init_accumulator` at the class level with `torch.cat`-based
implementations that build the growing accumulator slice without any storage
aliasing or `InPlaceSetSlice` references. These replacements are fully visible
to dynamo's graph tracer.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    50.15s
- Tier A attempts: N/A

## Files changed
- `babylm_baseline_gpt_bert_mixed/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a1f32e8e3570f2b3a321a84cb386f11bb2e5c91e |
| tt-forge-models | b827a34643713bd5477967c77aca265900dbb39a |
