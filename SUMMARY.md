# Remediation Summary: glm_4_7_trashflash_think_sorete_1b_test_i1_gguf-causal_lm-pytorch-Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_trashflash_think_sorete_1b_test_i1_gguf/causal_lm/pytorch-Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — PCC=0.782 for Gemma3 26-layer model on WH n150; ttmlir-bf16-matmul-precision-floor (Tier B)

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-12, 11], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})

Original traceback points to:
  transformers/cache_utils.py:214: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause

Three bugs were uncovered in sequence:

**Bug 1 (loader):** Before reaching the slice error, the test first failed with
`TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.
26 GGUF loaders in tt_forge_models monkey-patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
with the narrow signature `(gguf_path, return_tensors=False)`. transformers 5.2.0 now calls it with
`model_to_load=dummy_model`, causing `TypeError` on any GGUF model loaded after one of these loaders
was imported during pytest collection.

**Bug 2 (loader):** `mradermacher` GGUF tokenizers do not embed a `chat_template`. The loader called
`apply_chat_template` unconditionally, which raises an error for tokenizers without a template.

**Bug 3 (tt-xla):** After fixing the loader bugs, the slice error surfaced. PyTorch eager silently
clamps out-of-bounds negative slice indices (e.g. `t[:, :, -999:, :]` on a 12-element dim returns all
12 elements). The XLA lazy backend raises "Value out of range" instead. The Gemma3 SlidingWindowCache
computes `full_value_states[:, :, -sliding_window + 1 :, :]` where `sliding_window=512` but the
accumulated key/value tensor only has 12 elements in that dim during the first forward pass.

**Bug 4 (tt-mlir, Tier B):** After all three fixes, the test reaches compilation and execution but
produces PCC=0.782 vs the required 0.99. The model is Gemma3-architecture with 26 layers,
hidden_size=1152, and intermediate_size=6912. This PCC is consistent with the known WH BF16
matmul precision floor seen across Gemma, Qwen3, and GPT-J families: GPT-J 6B (28 layers,
intermediate=16384) gives PCC=0.75; Gemma 7B (32 layers) gives PCC~0.915; Gemma3 1B (26 layers,
intermediate=6912) at PCC=0.782 falls in the same band. Both TT and CPU operate on the same
BF16-dequantized Q4_K_M weights, so quantization noise is common; the divergence is TT-hardware-specific.

## Fix

**Bug 1 — tt_forge_models (`remediation/glm_4_7_trashflash_think_sorete_1b_test_i1_gguf-causal_lm-pytorch-Q4_K_M_GGUF-single_device-inference`, commit 83779c8bc3):**
Changed all 26 GGUF loaders from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    ...
    return _orig_load_gguf_checkpoint(*args, **kwargs)
```

**Bug 2 — tt_forge_models (`remediation/...`, commit db909d7e8c):**
Guarded `apply_chat_template` in `glm_4_7_trashflash_think_sorete_1b_test_i1_gguf/causal_lm/pytorch/loader.py`:
```python
if getattr(self.tokenizer, "chat_template", None) is not None:
    text = self.tokenizer.apply_chat_template(...)
else:
    text = self.sample_text
```

**Bug 3 — tt-xla (`remediation/glm_4_7_trashflash_think_sorete_1b_test_i1_gguf-causal_lm-pytorch-Q4_K_M_GGUF-single_device-inference`, commit b595dcf15):**
Added a pre-clamp guard in `TorchFunctionOverride.__torch_function__` in
`python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor`,
reads `size = tensor.shape[dim]` and clamps `start = max(start, -size)` and
`end = max(end, -size)` for statically-known dimension sizes.

**Bug 4 — proposed fix:**
The `ttmlir-bf16-matmul-precision-floor` would require enabling FP32 accumulation for all
BF16 matmuls in the tt-mlir lowering pipeline. This is a cross-cutting change affecting every
BF16 model and requires coordinated changes to the math fidelity settings throughout the
compilation stack.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting — the BF16 matmul precision floor requires enabling FP32 accumulation globally
across all BF16 lowerings in tt-mlir/tt-metal; it cannot be scoped to this one model or
one op pattern.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    334.90s (5m49s)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/glm_4_7_trashflash_think_sorete_1b_test_i1_gguf/causal_lm/pytorch/loader.py` (chat_template guard)
- 26 GGUF loader files in `tt_forge_models/` (narrow signature fix)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (aten.slice clamp)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f453f8187926e284c14130b5ade0c69fbc445535 |
| tt-forge-models | db909d7e8c9de3ad9bbb58de9427bfd45b7caa4e |
