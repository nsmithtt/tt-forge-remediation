# Remediation Summary: chemllm-causal_lm-pytorch-7B-Chat-1.5-DPO-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chemllm/causal_lm/pytorch-7B-Chat-1.5-DPO-single_device-inference]

## Result
FAIL — TT inference produces NaN logits (PCC=nan) after loader and Tier A compiler fixes; root cause of NaN is unknown

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
ttmlir-nan-output-internlm2-inference

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
E   RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: prims::view_of(Tensor(a) a) -> Tensor(a). We only support functionalizing operators whose outputs do not have alias annotations (e.g. 'Tensor(a)' is a Tensor with an alias annotation whereas 'Tensor' is a Tensor without. The '(a)' is the alias annotation). The alias annotation specifies that the output Tensor shares storage with an input that has the same annotation. Please check if (1) the output needs to be an output (if not, don't return it), (2) if the output doesn't share storage with any inputs, then delete the alias annotation. (3) if the output indeed shares storage with an input, then add a .clone() before returning it to prevent storage sharing and then delete the alias annotation. Otherwise, please file an issue on GitHub.
```

Remaining failure after both fixes:
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.
```

## Root cause

Three distinct issues were found:

**Issue 1 (loader, fixed):** transformers 5.x `validate_rope()` now requires a `factor` key when `rope_type='dynamic'`. ChemLLM's `InternLMConfig.__init__` defaults `rotary` to `{"base": 10000, "type": "dynamic"}` without `factor`, and the model's `config.json` also has `rope_scaling: {"base": 1000000, "type": "dynamic"}` without `factor`. Both injection points needed patching. Fixed in the loader via `_patch_internlm_config()` which intercepts `InternLMConfig.__init__` using `get_class_from_dynamic_module` and injects `factor=1.0` into both the `rotary` parameter and the `rope_scaling` kwarg path.

**Issue 2 (tt-xla, Tier A, fixed):** After the loader fix, the originally reported `prims::view_of` error surfaced. `squeeze` on tensors decomposes via `prims.squeeze + prims.view_of` in the FX graph. The `prims::view_of` op carries alias annotations (`Tensor(a) -> Tensor(a)`) that `partition_fx_graph_for_cpu_fallback`'s functionalization layer rejects. Since `view_of` is semantically an identity operation (same storage, same shape), replacing each node with its input is correct for inference. Fixed by adding `bypass_prims_view_of` FX pass in `tt_torch/backend/passes.py`.

**Issue 3 (tt-xla or tt-mlir, Tier B, unfixed):** After both fixes, the test runs to completion (105 seconds) but TT inference produces NaN logits. CPU golden output is valid (shape `[1, 9, 92544]`, finite values). The NaN propagates to PCC=nan. The exact source of NaN in the TT computation graph is unknown — it could be a precision issue accumulating across 32 transformer layers in bf16, or a specific op that returns NaN on TT hardware that has no diagnosed lowering bug. Diagnosing the NaN source would require additional instrumentation (per-layer output inspection on TT hardware) beyond the scope of a single Tier A fix.

## Fix

**Loader fix** (`tt_forge_models`, branch `remediation/chemllm-causal_lm-pytorch-7B-Chat-1.5-DPO-single_device-inference`):
- `chemllm/causal_lm/pytorch/loader.py`: Added `_patch_internlm_config()` method and import of `get_class_from_dynamic_module`. Called before `AutoModelForCausalLM.from_pretrained` in `load_model()`.

**Tier A compiler fix** (`tt-xla`, branch `remediation/chemllm-causal_lm-pytorch-7B-Chat-1.5-DPO-single_device-inference`):
- `python_package/tt_torch/backend/passes.py`: Added `bypass_prims_view_of()` FX pass.
- `python_package/tt_torch/backend/backend.py`: Added import and call of `bypass_prims_view_of` after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.

**Proposed fix for NaN (not attempted):** Instrument per-layer TT inference output to find first NaN, then determine whether it originates from a specific op lowering (potentially Tier A) or from accumulated bf16 precision loss across multiple layers (Tier B cross-cutting).

## Tier B justification

`internal-error-unknown-mechanism`: The TT inference produces NaN output, but without per-layer diagnostic instrumentation on TT hardware, the exact op or accumulation path that generates the NaN cannot be identified. Diagnosing the root cause requires additional infrastructure (layer-by-layer TT execution and comparison) that exceeds a single-file Tier A fix. Additionally, a second Tier A compiler-stack fix cannot be chained per skill rules (one compiler-stack fix per report).

## Verification
- pytest exit: FAIL
- Hardware: wormhole
- Duration: 105.42s (0:01:45)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py` — added `bypass_prims_view_of` FX pass
- `tt-xla/python_package/tt_torch/backend/backend.py` — import and call `bypass_prims_view_of`
- `tt-xla/third_party/tt_forge_models/chemllm/causal_lm/pytorch/loader.py` — `_patch_internlm_config` for transformers 5.x RoPE factor requirement

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 929f90bf1b6f1e48b50c676905070b7a91b923ff |
| tt-forge-models | e8a65aa7c1f9078c9cf19e20dd9d7bae6ddf9573 |
