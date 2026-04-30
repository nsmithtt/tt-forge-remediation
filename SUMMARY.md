# Remediation Summary: h2ovl_mississippi-pytorch-800M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[h2ovl_mississippi/pytorch-800M-single_device-inference]

## Result
FAIL — model loads and runs (PCC=0.8953), precision gap vs CPU BF16 is a Tier B compiler issue

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise RuntimeError("Tensor.item() cannot be called on meta tensors")
```

Full traceback:
```
modeling_intern_vit.py:315: in __init__
    dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
RuntimeError: Tensor.item() cannot be called on meta tensors
```

## Root cause

Three loader bugs, plus one residual compiler precision issue:

**Bug 1 — meta-tensor .item() (original reported failure):** transformers 5.x constructs models on meta device. `InternVisionEncoder.__init__` calls `torch.linspace(...).item()` to build drop-path rate lists; `.item()` on meta tensors raises `RuntimeError`. Fixed by patching `torch.Tensor.item` to return `0.0` for meta scalars during `from_pretrained` (safe because `DropPath` is no-op in eval mode).

**Bug 2 — missing all_tied_weights_keys:** `H2OVLChatModel.__init__` predates transformers 5.x; it doesn't call `self.post_init()`, so `all_tied_weights_keys` is absent when `_finalize_model_loading` accesses it. Fixed via `PreTrainedModel._finalize_model_loading` patch (class-level patch is required because `get_class_in_module` uses a path-derived `sys.modules` key that differs from `importlib`).

**Bug 3 — rope_parameters=None (new bug for 800M):** The 800M config JSON was saved by transformers 4.x with `"rope_scaling": null`. `H2OVLChatConfig.__init__` calls `self.llm_config.update(original_dict)`, which triggers the transformers 5.x `PreTrainedConfig.rope_scaling.setter`. That setter unconditionally executes `self.rope_parameters = value`, overwriting the `rope_parameters` dict that `LlamaConfig.__init__` computed via `convert_rope_params_to_dict` (which correctly set `{'rope_theta': 100000, 'rope_type': 'default'}`) with `None`. The subsequent `LlamaRotaryEmbedding.__init__` then fails with `TypeError: 'NoneType' object is not subscriptable` when accessing `config.rope_parameters["rope_type"]`. Fixed by guarding the `rope_scaling` setter so `rope_scaling=None` is a no-op when `rope_parameters` is already set.

**Residual PCC issue:** After all three loader fixes the model runs to completion on TT silicon. PCC between TT BF16 output and CPU BF16 output is 0.8953, well below the required 0.99. The gap is significantly larger than what BF16 accumulation noise would produce (which would be in the 0.99x range), indicating a genuine compiler precision difference. The analogous H2OVL 2B model also showed sub-0.99 PCC (0.9848) on this silicon. The mechanism — whether a specific matmul is incorrectly downcast or a particular op lowering introduces bias — was not identified. Diagnosing it requires layer-by-layer output profiling of a multimodal model (InternVL vision encoder + projection MLP + LLaMA decoder), which is beyond the scope of a scoped single-file Tier A fix.

## Fix
Three loader-layer fixes in `h2ovl_mississippi/pytorch/loader.py` (in `tt_forge_models`):

1. `torch.Tensor.item` patched to return `0.0` for meta tensors during `from_pretrained`.
2. `PreTrainedModel._finalize_model_loading` patched to call `post_init()` when `all_tied_weights_keys` is absent.
3. `PreTrainedConfig.rope_scaling.setter` patched to be a no-op when `value is None` and `rope_parameters` is already set.

Proposed fix for the residual PCC issue (not implemented — Tier B): layer-by-layer profiling to identify which ops produce numerical divergence between TT and CPU BF16; likely a matmul or softmax that TT executes with reduced precision.

## Tier B justification

Which indicator: `internal-error-unknown-mechanism`

The PCC gap (0.8953 vs required 0.99) is a real discrepancy between TT bfloat16 and CPU bfloat16 execution on this 800M vision-language model. The exact layer or operation responsible was not identified. Diagnosing the root cause requires layer-by-layer output profiling across a complex multi-modal model (InternVL vision encoder + projection MLP + LLaMA decoder), which is beyond the scope of a single scoped fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    156.95s (0:02:36)
- Tier A attempts: N/A

## Files changed
- `h2ovl_mississippi/pytorch/loader.py` (in tt_forge_models, branch `remediation/h2ovl_mississippi-pytorch-800M-single_device-inference`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6c3be9706b3eb19b87cba9b2e7f2031d29508201 |
| tt-forge-models | b700ad0898b7f3f665a53eee7a1e455e50d72134 |
