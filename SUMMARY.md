# Remediation Summary: jina_clip_v1-pytorch-jina-clip-v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jina_clip_v1/pytorch-jina-clip-v1-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-meta-device-non-persistent-buffer-nan

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Tensor.item() cannot be called on meta tensors

The error originated in EVAVisionTransformer.__init__ (eva_model.py:606):
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

## Root cause
Transformers 5.x unconditionally initializes models under a meta device context.
Three distinct non-persistent buffer initialization bugs were found in the loader:

1. **EVAVisionTransformer (vision model)**: `torch.linspace(0, drop_path_rate, depth).item()`
   is called in `__init__` to compute stochastic-depth rates. Under meta device context,
   `linspace` returns a meta tensor and `.item()` raises RuntimeError.

2. **VisionRotaryEmbeddingFast (vision model)**: `freqs_cos` and `freqs_sin` are
   non-persistent buffers computed from `torch.arange` under the meta context. After
   weight loading, these buffers are absent from the checkpoint and get uninitialized
   (NaN) values, corrupting vision encoder outputs.

3. **JinaBERT SelfAttention (text model)**: `alibi_slopes` (ALiBi positional biases)
   is computed via `torch.tensor(get_alibi_slopes(num_heads), device=meta_device)` under
   meta context. After materialization, the buffer contains garbage values. `linear_biases`,
   derived by multiplying garbage slopes with position distances, overflows to NaN in
   bfloat16, poisoning all text encoder outputs.

Additionally: (a) `load_dataset` in `load_inputs` triggered a spacy/dill namespace
collision, replaced with PIL.Image + get_file; (b) the text tower loads with a different
dtype (float16/float32) than the vision tower (bfloat16) due to nested config override;
(c) `CLIPOutput` (dict-like model output) was replaced with return_dict=False tuple output
to avoid PCC comparison failures from nested None fields.

## Fix
All fixes are in `jina_clip_v1/pytorch/loader.py` in the tt-forge-models repo on branch
`remediation/jina_clip_v1-pytorch-jina-clip-v1-single_device-inference`.

**Commit 1** — EVA meta device linspace fix:
- Temporarily replace `torch.linspace` with a CPU-forcing wrapper during `from_pretrained`
- Add `_recompute_rope_buffers()`: recompute `freqs_cos/freqs_sin` non-persistent buffers
  from config parameters after loading
- Add `_fix_eva_rope_forward_accumulation()`: replace `EVAVisionTransformer.forward_features`
  with a stable-partial version to prevent Dynamo recompilation past its limit

**Commit 2** — Replace `load_dataset` with `PIL.Image` + `get_file` to avoid spacy/dill
namespace collision crash.

**Commit 3** — Add `return_dict=False` to model kwargs to force tuple output for clean PCC
comparison (avoids None fields in nested CLIPOutput).

**Commit 4** — Add `model.to(dtype=dtype_override)` after loading to unify text/vision
tower dtypes (text tower config overrides dtype to float16/float32 via nested _from_config).

**Commit 5** — Add `_recompute_alibi_buffers()`: recompute JinaBERT `alibi_slopes` and
`linear_biases` non-persistent buffers. Reimplements `get_alibi_slopes(nheads)` in pure
Python to avoid importing the dynamic remote module, then calls `_build_linear_biases(16)`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    61.23s (0:01:01)
- Tier A attempts: N/A

## Files changed
- tt-forge-models: `jina_clip_v1/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 723400d6fcee7f9e55f995c46642c902a807dfed |
