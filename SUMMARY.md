# Remediation Summary: bilingual_embedding_large/embedding_generation/pytorch-Lajavaness/bilingual-embedding-large-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bilingual_embedding_large/embedding_generation/pytorch-Lajavaness/bilingual-embedding-large-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-uninit-token-type-ids-buffer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
IndexError: index out of range in self

Full traceback:
  tests/infra/testers/single_chip/model/model_tester.py: _run_on_cpu
    return self._device_runner.run_on_cpu(compiled_workload)
  transformers_modules/dangvantuan/bilingual_impl/.../modeling.py:108
    token_type_embeddings = self.token_type_embeddings(token_type_ids)
  torch/nn/functional.py:2542
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
  IndexError: index out of range in self

Reported failure message: sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

## Root cause
Loader bug in tt_forge_models. The BilingualEmbeddings module registers
`token_type_ids` as a non-persistent buffer initialized to zeros. In
transformers 5.x, `from_pretrained` uses `init_empty_weights` (meta device)
during model construction; non-persistent buffers are not written by the
checkpoint and are never re-materialized, leaving them with garbage values
after loading. The `AutoTokenizer` for `Lajavaness/bilingual-embedding-large`
does not return `token_type_ids` in its output, so `BilingualEmbeddings.forward`
falls back to `self.token_type_ids` (the uninitialized buffer). The garbage
values cause an `IndexError` when the embedding matrix lookup executes — the
embedding table has only 2 token types (0 and 1) but the uninitialized buffer
contains out-of-range integers.

The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__
attribute` in the reported failure message is an unrelated warning from the
UMD SWIG bindings emitted at Python startup; it is not the cause of failure.

## Fix
Added zeroing of `token_type_ids` buffers in `load_model()` after
`AutoModel.from_pretrained()`:

```python
for module in model.modules():
    if hasattr(module, "token_type_ids"):
        module.token_type_ids.zero_()
```

File changed: `bilingual_embedding_large/embedding_generation/pytorch/loader.py`
in `tt_forge_models`, on branch
`remediation/bilingual-embedding-large-embedding-generation-pytorch-bilingual-embedding-large-single-device-inference`,
commit `ce5e2b4f46499ecea39577a71de7f52c00894af1`.

The same fix was previously applied to the `bilingual-embedding-small` model
(commit `6d845d8eab`).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    89.31s (0:01:29)
- Tier A attempts: N/A

## Files changed
- `bilingual_embedding_large/embedding_generation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ab18b7caa8f648b461cd1e8619c7285f5743a8c6 |
| tt-forge-models | ce5e2b4f46499ecea39577a71de7f52c00894af1 |
