# Remediation Summary: darkidol_catgirl_gguf/causal_lm/pytorch-9B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[darkidol_catgirl_gguf/causal_lm/pytorch-9B_GGUF-single_device-inference]

## Result
FAIL — Darkidol-Catgirl-9B is Qwen3.5-9B (qwen35 arch, hybrid SSM+full-attention); transformers has no GGUF→Qwen3Next loading path, causing size mismatches in full-attention layer projections

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-no-transformers-tensor-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing the `_patched_load_gguf_checkpoint` kwargs bug (Bug 1), the test fails with:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

The mismatch report shows:
```
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH | ckpt: torch.Size([1024, 4096]) vs model: torch.Size([512, 4096])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH | ckpt: torch.Size([8192, 4096]) vs model: torch.Size([2048, 4096])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.v_proj.weight | MISMATCH | ckpt: torch.Size([1024, 4096]) vs model: torch.Size([512, 4096])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_norm.weight | MISMATCH | ckpt: torch.Size([256]) vs model: torch.Size([128])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_norm.weight | MISMATCH | ckpt: torch.Size([256]) vs model: torch.Size([128])
```

## Root cause

Two bugs found in sequence:

**Bug 1 (fixed):** The `_patched_load_gguf_checkpoint` functions in 26 qwen35 GGUF loaders did not accept `**kwargs`. When pytest collects all tests, these loaders are imported as module-level side effects, replacing the global `load_gguf_checkpoint` in transformers. In transformers 5.2.0, `load_gguf_checkpoint` gained a `model_to_load` keyword argument. The global monkey-patch intercepted this call and raised `TypeError`, which is the `raise AttributeError(` failure reported.

**Bug 2 (Tier B):** The Darkidol-Catgirl-9B model is based on `Qwen/Qwen3.5-9B`, a hybrid SSM+full-attention architecture (`general.architecture: qwen35`, `qwen35.full_attention_interval: 4`). The qwen35 loaders' patch maps `model_type=qwen35` → `model_type=qwen3`, causing `Qwen3ForCausalLM` to be used — which creates all layers with uniform attention dims (`num_attention_heads=16, head_dim=128`). Full-attention layers at indices 3, 7, 11, ..., 31 need `num_attention_heads=32, head_dim=256`, producing a 2–4× size mismatch in attention projection weights.

The correct model class is `Qwen3NextForCausalLM` (`model_type=qwen3_next`), which understands hybrid `layer_types` via `full_attention_interval`. However, transformers has no GGUF→Qwen3Next loading path: `qwen35` is absent from `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`, and no config-key or tensor-name translation exists for the hybrid layers.

## Fix

**Bug 1 (applied):** In `tt_forge_models`, on branch `remediation/darkidol_catgirl_gguf_causal_lm_pytorch_9B_GGUF_single_device_inference` (commit `fc9699ea7d`):

Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and forwarded `**kwargs` to `_orig_load_gguf_checkpoint(...)` in all 26 affected loader files.

**Bug 2 (proposed fix):** Add `qwen35` to `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` in `transformers/modeling_gguf_pytorch_utils.py`, mapping to `qwen3_next` model_type with correct config key mappings for `full_attention_interval`, `num_attention_heads` (derived from full-attention layer tensor shapes, not GGUF metadata which describes linear-attention dims), and `ssm.*` linear attention params. Additionally, GGUF tensor names for hybrid blocks need mapping to `Qwen3NextForCausalLM` PyTorch parameter names.

## Tier B justification
**Indicator: new-infrastructure**

Transformers has no GGUF loading infrastructure for the qwen35 hybrid architecture or qwen3_next: both `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` lack entries for these architectures, and the GGUF tensor name → PyTorch parameter name translation needed for hybrid layers (separate linear-attention and full-attention layers per block) does not exist in any existing transformer GGUF loader.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    419.15s (0:06:59) for run with Bug 1 fix applied, revealing Bug 2
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/*/causal_lm/pytorch/loader.py` (26 files): add `**kwargs` to `_patched_load_gguf_checkpoint` (on remediation branch `remediation/darkidol_catgirl_gguf_causal_lm_pytorch_9B_GGUF_single_device_inference`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | fc9699ea7daba758450dc02b3b2280464443b9fc |
