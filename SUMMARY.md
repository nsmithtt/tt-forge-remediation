# Remediation Summary: mozilla_ai_meta_llama_3_1_70b_instruct_llamafile-causal_lm-pytorch-Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile-tensor_parallel-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch-Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile-tensor_parallel-inference]

## Result
FAIL — loader bug fixed (GGUF llamafile parsing + broken chain bypass); cannot verify SILICON_PASS on single-chip p150b because tensor_parallel requires multi-chip mesh

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-llamafile-load-chain-bypass

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
2026-04-23 21:17:49.902 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```
On our p150b hardware, the real failure is:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
which occurs because `AutoModelForCausalLM.from_pretrained` with `gguf_file=` passes `model_to_load=dummy_model` (transformers 5.x), but the patched function installed by other GGUF loaders in the test session has no `**kwargs`. After fixing the chain bypass, the test fails with:
```
AssertionError: Tensor parallel requires multi-chip mesh
```

## Root cause
Two independent loader bugs:

**Bug 1 — GGUF magic invalid**: A `.llamafile` file is an APE polyglot (ZIP + executable binary). The 40 GB GGUF weights are stored as an *uncompressed* ZIP entry at byte offset 242,745,344. The loader passed the `.llamafile` path directly to `GGUFReader` which calls `np.memmap(path, offset=0)` and reads APE executable header bytes instead of the GGUF magic.

Fix: `_find_gguf_offset_in_llamafile()` reads the ZIP local file header to compute the exact data offset, and `_gguf_from_llamafile_ctx()` patches `gguf.gguf_reader.np.memmap` to inject `offset=242745344` for the specific file path.

**Bug 2 — broken load_gguf_checkpoint chain**: 26+ GGUF loaders in the test suite each install a patch on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. Most broken patches have signature `(gguf_path, return_tensors=False)` — no `**kwargs`. Transformers 5.x adds `model_to_load=dummy_model` to the call at `modeling_utils.py:4016`. The last-installed broken patch (from another loader imported after ours) rejects this with `TypeError`.

The broken patches store their predecessors in module-level variables (not closures), so simple closure traversal fails to find the real transformers function. Fix: `_find_real_load_gguf_checkpoint()` uses BFS through both closures and `__globals__` (scanning variables with common predecessor-naming prefixes: `_orig*`, `orig_*`, `_chain*`, etc.) and uses `fn.__module__ == 'transformers.modeling_gguf_pytorch_utils'` as the primary criterion for identifying the real function. At call time in `load_model`/`load_config`, we temporarily install `_real_load_gguf_checkpoint` and restore the previous function in a `try/finally`.

**Hardware constraint**: After both loader fixes, the 70B model loads successfully (724 GGUF tensors dequantized to BF16 in ~22 minutes). The test then hits `AssertionError: Tensor parallel requires multi-chip mesh` because tensor_parallel requires ≥2 devices and our p150b has 1. The 70B Q4_K_M model dequantizes to ~140 GB BF16, requiring n300-llmbox or galaxy-wh-6u for tensor parallel.

## Fix
**tt_forge_models** (`mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch/loader.py`):
- Added `_find_gguf_offset_in_llamafile()`: reads ZIP local file header to get GGUF data start offset
- Added `_gguf_from_llamafile_ctx()`: context manager that patches `gguf.gguf_reader.np.memmap` with correct offset
- Added `_find_real_load_gguf_checkpoint()`: BFS through closures + `__globals__` (predecessor name prefixes + `__module__` check) to find the real transformers function
- `load_model`/`load_config`: temporarily install `_real_load_gguf_checkpoint` around `from_pretrained` calls

**tt-xla** (`tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`):
- Added `mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch-...` with `supported_archs: [n300-llmbox, galaxy-wh-6u]` and `status: NOT_STARTED` so the test is properly skipped on single-chip hardware instead of raising AssertionError.

## Verification
- pytest exit: FAIL (AssertionError: Tensor parallel requires multi-chip mesh)
- Hardware:    p150b (single chip — insufficient for tensor_parallel)
- Duration:    1331.98s (0:22:11) — model load succeeded; failure at mesh setup
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch/loader.py`
- `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f13d8d76f985870e9a88368dae7dbe0bda443910 |
| tt-forge-models | 63124f056a2800206b87f914d73510b8be26ada4 |
