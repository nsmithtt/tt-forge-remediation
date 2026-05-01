# Remediation Summary: lavida-image_text_to_text-pytorch-lavida_llada_v1_0_instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lavida/image_text_to_text/pytorch-lavida_llada_v1_0_instruct-single_device-inference]

## Result
FAIL — PCC=0.921 below required 0.99; suspected ttmlir-bf16-matmul-precision-floor for ~7B multimodal model

## Stack layer
loader, tt-xla, tt-mlir

## Tier
A

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
```
AttributeError: 'fused_0' object has no attribute 'xla_args'
```
After loader fixes, the test advanced to a compiler-stack error. After the Tier A fix, the test advances to:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9210424996823333. Required: pcc=0.99.
```

## Root cause

**Loader bugs (tt_forge_models):** The model class `LlavaLladaForMaskedDiffusion.__init__` calls `SigLipVisionModel.from_pretrained()` during `__init__`, which is forbidden inside transformers 5.x's `init_empty_weights()` meta-device context. Additional loader bugs: `tie_weights` signature incompatibility, `SigLipVisionConfig._set_token_in_kwargs` removal, non-persistent `position_ids` buffer left uninitialised after `load_model()`, activation checkpointing using `torch.checkpoint` with `use_reentrant=False` which calls `torch.xla` attribute path raising `AttributeError`, non-standard output fields in `CausalLMOutputWithPast` breaking pytree unflattening, and missing `labels` tensor for the masked-diffusion forward.

**tt-xla Tier A bug:** The lavida forward pass has a dynamo graph break (triggered by in-place mutations on `attention_mask`/`labels` before the main transformer). The resulting pre-processing subgraph returns no tensor outputs (empty `output_specs`). `partition_fx_graph_for_cpu_fallback` groups the in-place XLA ops into `fused_0`, but since the graph's output node has no data dependency on `fused_0`, `legalize_graph` places the output node before `fused_0` in topological order. `Interpreter.run()` terminates at the output node and never visits `fused_0`, so `InputCollector` never executes `call_module("fused_0", …)` and never sets `fused_0.xla_args`. The subsequent `extract_graph_helper` access to `xla_model.xla_args` raises `AttributeError`.

**Remaining PCC bug:** After fixing both the loader and the tt-xla backend, the model runs to completion and produces PCC=0.921. This is consistent with the known Wormhole BF16 matmul precision floor seen in other ~7B-parameter models (Gemma 7B: 0.915, GPT-J 6B: 0.75, InternLM3 8B: 0.272). The LLaDA backbone is a masked-diffusion LM with large MLP projections; BF16 accumulation in the many-layer transformer compounds the rounding error.

## Fix

**Loader fixes** (tt_forge_models, `lavida/image_text_to_text/pytorch/loader.py`):
1. Set `config.delay_load = True` so `SigLipVisionTower.__init__` defers `from_pretrained` outside the meta-device context; call `vision_tower.load_model()` explicitly after `AutoModelForCausalLM.from_pretrained`.
2. Shim `LLaDAModelLM.tie_weights` to accept `missing_keys`/`recompute_mapping` kwargs (transformers 5.x calling convention).
3. Shim `LLaDAModelLM.forward` to accept and ignore `position_ids`, `prompt_len`, `num_items_in_batch` kwargs injected by transformers 5.x.
4. Add `SigLipVisionConfig._set_token_in_kwargs` no-op (removed in transformers 5.x).
5. Re-initialise non-persistent `position_ids` buffer in `SigLipVisionEmbeddings` after `load_model()` (left on meta device because it is not in the checkpoint).
6. Replace activation-checkpointing function with a plain call (PyTorch 2.7's `use_reentrant=False` path tries `getattr(torch, "xla")` which fails).
7. Wrap `LlavaLladaForMaskedDiffusion.forward` to return a plain `CausalLMOutputWithPast` without non-standard fields (`new_input_ids`, `labels`, `final_masked_indices`, `p_mask`).
8. Provide `labels` tensor (prompt=-100, response=MASK_ID) required by the masked-diffusion forward.
9. Use `vision_tower.image_processor.preprocess()` instead of `__call__` (no `__call__` on `SigLipImageProcessor`).

**Tier A fix** (tt-xla, `python_package/tt_torch/backend/backend.py`):
When `program.graph_signature.output_specs` is empty, skip `extract_compiled_graph` entirely; store `_EMPTY_OUTPUT_SENTINEL` as `self.compiled_graph` and dispatch directly via `self.module(*args)` on each call (XLA lazy-evaluation path handles the in-place mutations).

**Proposed fix for remaining PCC** (tt-mlir):
The WH BF16 matmul accumulation floor requires higher-precision intermediate arithmetic (F32 accumulation or math-fidelity increase) for large models. This is a cross-cutting compiler change.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    720.98s (0:12:00)
- Tier A attempts: 1

## Files changed
- `tt_forge_models` `lavida/image_text_to_text/pytorch/loader.py` — loader fixes (1–9 above)
- `tt-xla` `python_package/tt_torch/backend/backend.py` — empty-output-specs bypass
- `tt-xla` `third_party/tt_forge_models` — submodule pointer updated to `94db546833`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2e0d6d9a4e07357f27f262d0bb4beba446eb0636 |
| tt-forge-models | 94db546833aa26bff1dc249deff17e11d046f2b8 |
