# Remediation Summary: bilingual-embedding-small-embedding-generation-pytorch-bilingual-embedding-small-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bilingual_embedding_small/embedding_generation/pytorch-bilingual-embedding-small-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-pretrained-config-attrs-removed-and-uninit-buffers

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'BilingualConfig' object has no attribute 'is_decoder'
(followed by add_cross_attention missing, return_dict kwarg leak, and
IndexError: index out of range in self in token_type_embeddings)

## Root cause
Three distinct transformers 5.x breaking changes, all in the loader layer:

1. **Missing PretrainedConfig defaults** (`is_decoder`, `add_cross_attention`):
   `BilingualConfig` inherits from `PretrainedConfig` but does not explicitly
   set `is_decoder` or `add_cross_attention`. Older transformers provided these
   as defaults on `PretrainedConfig`; transformers 5.x removed them. The model
   `__init__` reads them from config and raises `AttributeError`.

2. **`return_dict=False` leaks into model `__init__`**: When a pre-built config
   object is passed to `AutoModel.from_pretrained`, transformers 5.x no longer
   extracts known config attributes from `**kwargs`, so `return_dict=False` ends
   up in `model_kwargs` at the `cls(config, ...)` call site, causing
   `TypeError: BilingualModel.__init__() got an unexpected keyword argument 'return_dict'`.

3. **Uninitialized non-persistent buffer** (`token_type_ids`): transformers 5.x
   uses `init_empty_weights` (meta device) during `from_pretrained`. The
   `BilingualEmbeddings.__init__` registers `token_type_ids` as
   `torch.zeros(..., persistent=False)`. This buffer is not in the checkpoint,
   so after materialization it contains garbage memory (e.g. 63613825772945407),
   causing `IndexError: index out of range in self` in `token_type_embeddings`
   during the CPU forward pass.

## Fix
File: `bilingual_embedding_small/embedding_generation/pytorch/loader.py` in
`tenstorrent/tt-forge-models`, remediation branch
`remediation/bilingual-embedding-small-embedding-generation-pytorch-bilingual-embedding-small-single-device-inference`.

Changes:
- Import `AutoConfig` alongside `AutoModel`.
- Pre-load config via `AutoConfig.from_pretrained(…, trust_remote_code=True)`.
- Patch `config.is_decoder = False` and `config.add_cross_attention = False` if
  missing (guards against future transformers restoring them).
- Set `config.return_dict = False` directly on the config object and remove
  `return_dict` from `model_kwargs`.
- After `from_pretrained`, iterate all modules and call `.zero_()` on any
  `token_type_ids` buffer to reinitialize uninitialized meta-device residue.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    48.81s
- Tier A attempts: N/A

## Files changed
- `bilingual_embedding_small/embedding_generation/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 6d845d8eab4a1776450129c9cf0052d4bc0095f4 |
