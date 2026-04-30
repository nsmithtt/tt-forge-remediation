# Remediation Summary: bert-mesh_classification-pytorch-marcmendez_aily_BertMeshTerms-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert/mesh_classification/pytorch-marcmendez_aily_BertMeshTerms-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-nested-from-pretrained-meta-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: You are using `from_pretrained` with a meta device context manager or `torch.set_default_device('meta')`.
This is an anti-pattern as `from_pretrained` wants to load existing weights.
If you want to initialize an empty model on the meta device, use the context manager or global device with `from_config`, or `ModelClass(config)`
```

## Root cause
`BertMeshModel` (remote code at `marcmendez-aily/BertMeshTerms`) calls
`AutoModel.from_pretrained(self.pretrained_model)` inside its own `__init__`, and never
calls `self.post_init()`. Transformers 5.x introduced four incompatibilities with this
pattern, all in the loader layer:

1. **Meta-device context**: `PreTrainedModel.get_init_context` unconditionally appends
   `torch.device("meta")` to the model-init context managers, so the outer
   `from_pretrained` runs `BertMeshModel.__init__` inside a meta-device context. The
   inner `AutoModel.from_pretrained` call detects the meta device (via
   `check_and_set_device_map`) and raises `RuntimeError`.

2. **Missing `all_tied_weights_keys`**: `_finalize_model_loading` calls
   `_adjust_tied_keys_with_tied_pointers`, which accesses `model.all_tied_weights_keys`.
   This attribute is initialised by `post_init()`, which `BertMeshModel` never calls,
   so the attribute is absent.

3. **Non-persistent buffer corruption**: `_move_missing_keys_from_meta_to_device`
   unconditionally replaces every non-persistent buffer (including
   `bert.embeddings.position_ids` = `arange(512)`) with `torch.empty_like` (garbage
   memory), because the function assumes the model was initialised on meta device.
   Without meta, the correctly-seeded values are destroyed.

4. **Dtype mismatch**: The inner `from_pretrained` ("BiomedNLP-BiomedBERT-base-uncased-
   abstract") runs its own `local_torch_dtype(float32)` init context, so the inner BERT
   submodule ends up in float32. When the outer `convert_and_load_state_dict_in_model`
   runs with `dtype=bfloat16`, it sees `empty_param.dtype == float32 != bfloat16` and
   falls back to float32 for bert's weights (line 1162 of core_model_loading.py),
   producing a float32/bfloat16 dtype mismatch on the first `torch.matmul` in
   `MultiLabelAttention.forward`.

## Fix
All fixes are in `bert/mesh_classification/pytorch/loader.py` in
`tenstorrent/tt-forge-models` on branch
`remediation/bert-mesh_classification-pytorch-marcmendez_aily_BertMeshTerms-single_device-inference`
(commit `3bfea50b8f02ff5efa167aa418648d6bf324de29`).

In `load_model`, the outer `from_pretrained` call is wrapped with temporary patches to
`PreTrainedModel`:

- `get_init_context` is patched to filter out `torch.device("meta")` from the context
  list, allowing the inner `from_pretrained` to succeed on CPU.
- `_finalize_model_loading` is patched to (a) call `model.post_init()` if
  `all_tied_weights_keys` is absent, (b) save non-persistent buffer values before
  `_finalize_model_loading` runs and restore them after, so `position_ids` keeps its
  `arange(512)` seed.

After `from_pretrained` returns, `model.to(dtype_override)` is called to cast the
entire model (including the float32 BERT submodule) to the requested dtype.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    31.47s
- Tier A attempts: N/A

## Files changed
- `bert/mesh_classification/pytorch/loader.py` (in `tenstorrent/tt-forge-models`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3bfea50b8f02ff5efa167aa418648d6bf324de29 |
