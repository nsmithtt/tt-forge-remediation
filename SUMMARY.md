# Remediation Summary: glove_6b-embedding_generation-pytorch-NeuML-glove-6B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glove_6b/embedding_generation/pytorch-NeuML/glove-6B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
glove-6b-string-input-not-traceable-by-dynamo

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <Wrapped method <original div_>>(*(FakeTensor(..., size=(1, 300)),
FakeTensor(..., size=(1, 1))), **{}): got AttributeError("'ndarray' object has no
attribute 'div_'")

from user code:
   File "loader.py", line 45, in forward
    embeddings = self._sv.embeddings(input_texts)
  File "staticvectors/model.py", line 69, in embeddings
    self.normalize(embeddings)
  File "staticvectors/model.py", line 156, in normalize
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
  File "tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

The first-seen failure was `ModuleNotFoundError: No module named 'staticvectors'`
(no requirements.txt in the loader directory).

## Root cause
Two loader bugs:

1. **Missing requirements.txt**: The loader directory had no `requirements.txt`,
   so the `staticvectors` package was never declared as a dependency. The
   `RequirementsManager` therefore never installed it before the test ran.

2. **Non-traceable forward method**: The original `GloVeEmbeddingModel.forward`
   took `input_texts: list[str]` and called `self._sv.embeddings(input_texts)` —
   a numpy-only code path inside `staticvectors.model.StaticVectors`. TorchDynamo
   intercepted the `/=` (in-place div) inside `staticvectors.normalize()` via
   `tt_torch/torch_overrides.py TorchFunctionMode.__torch_function__`, but
   `np.linalg.norm(...)` returned an ndarray while Dynamo was propagating
   FakeTensors, causing the `AttributeError`.  String inputs are not tensor types,
   so Dynamo cannot trace the lookup at all.

## Fix
Both fixes are in the loader (`tt_forge_models/glove_6b/embedding_generation/pytorch/`).

1. **Added `requirements.txt`** with `staticvectors`.

2. **Restructured `GloVeEmbeddingModel`**: At init time, the 400K×300 float32
   weight matrix is extracted from `StaticVectors.vectors` and loaded into
   `nn.Embedding.from_pretrained(weights, freeze=True)`. The `forward` now
   accepts `token_ids: torch.Tensor` (integer indices) and performs:
   - `nn.Embedding` lookup → `[batch, seq_len, 300]`
   - L2 normalization: `emb / (emb * emb).sum(-1, keepdim=True).sqrt()`

   `ModelLoader.load_model` stores the `StaticVectors` instance on `self._sv`.
   `ModelLoader.load_inputs` splits the sample sentence into words, maps each
   word to its vocabulary index via `sv.tokens`, and returns a `torch.long`
   tensor of shape `[1, num_words]`.

- `tt_forge_models` commit `4c3266bdf6` on branch
  `remediation/glove_6b-embedding_generation-pytorch-NeuML-glove-6B-single_device-inference`
- `tt-xla` commit `334e98d34` on branch
  `remediation/glove_6b-embedding_generation-pytorch-NeuML-glove-6B-single_device-inference`
  (advances `third_party/tt_forge_models` pointer to `4c3266bdf6`)

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    40.25s
- Tier A attempts: N/A

## Files changed
- `glove_6b/embedding_generation/pytorch/requirements.txt` (new file)
- `glove_6b/embedding_generation/pytorch/loader.py` (restructured model)

## Submodule hashes
| Submodule       | Commit                                   |
|-----------------|------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 334e98d34481e4e8c0bcd8f7eb9fce8764fd7e79 |
| tt-forge-models | 4c3266bdf6dd1bfd300ff1ba63578874999eb909 |
