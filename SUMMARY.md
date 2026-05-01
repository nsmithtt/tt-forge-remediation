# Remediation Summary: ministral_8b_gguf-causal_lm-pytorch-lmstudio_Ministral-3-8B-Instruct-2512-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ministral_8b_gguf/causal_lm/pytorch-lmstudio_Ministral-3-8B-Instruct-2512-GGUF-single_device-inference]

## Result
FAIL — PCC 0.9524 on TT silicon vs required 0.99; TT BF16 matmul accumulation vs CPU FP32 accumulation causes cross-cutting precision gap across 34-layer model with Q4_K_M dequantized weights (Tier B)

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original:
```
raise NotImplementedError(
```
Full error: `NotImplementedError: Unknown gguf model_type: mistral` (from gguf-py `MODEL_ARCH_NAMES` lookup in `get_gguf_hf_weights_map`).

After initial loader fix, second error:
```
RuntimeError: Value out of range [-128, 127], but got -4095
```
in `aten.slice.Tensor` from `SlidingWindowLayer.update()` (MistralConfig defaults `sliding_window=4096`).

After both fixes applied, test runs to completion but PCC = 0.9524 vs required 0.99.

## Root cause
Three separate root causes were found and fixed, plus one Tier B remaining issue:

**Bug 1 (loader): `NotImplementedError: Unknown gguf model_type: mistral`**
`get_gguf_hf_weights_map` in transformers 5.x calls gguf-py with `hf_model.config.model_type` ("mistral"). gguf-py 0.18+ uses `"mistral3"` (not `"mistral"`) in `MODEL_ARCH_NAMES`. The loader had no registration for `mistral3` in GGUF tables and no `get_gguf_hf_weights_map` patch.

**Bug 2 (tt-xla, Tier A): `RuntimeError: Value out of range [-128, 127], but got -4095`**
After remapping `model_type → mistral`, `MistralConfig.__init__` defaults `sliding_window=4096`. With `sliding_window=4096` and a 128-token prefill, `SlidingWindowLayer.update()` generates `full_kv[:, :, -4095:, :]`  — a `start=-4095` argument to `aten.slice.Tensor` on dimension 2 of size 128. PyTorch eager silently clips this; XLA/TT validates strictly and raises `Value out of range`. Two fixes were applied: (a) set `sliding_window=None` in the loader patch to prevent the MistralConfig default from activating sliding window; (b) add `clamp_out_of_range_slice_starts` FX pass in tt-xla as a defensive Tier A fix for any similar models.

**Bug 3 (loader): `ValueError: Unrecognized configuration class Mistral3Config`** (intermittent)
`configuration_utils.py`, `tokenization_utils_tokenizers.py`, and `tokenization_auto.py` all do module-level `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint`, binding the original function before the loader's monkey-patch runs. The patch must explicitly overwrite those bindings after patching the module attribute.

**Bug 4 (loader): `KeyError: 'mistral'` in `convert_gguf_tokenizer`**
The GGUF file's `tokenizer.ggml.model` field is `"mistral"` (not `"mistral3"`). `convert_gguf_tokenizer` looks up this key in `GGUF_TO_FAST_CONVERTERS`. Adding `GGUF_TO_FAST_CONVERTERS["mistral"] = GGUFLlamaConverter` was required.

**Tier B — PCC 0.9524 (ttmlir-bf16-precision-not-preserved)**
After all loader and Tier A fixes, the test completes but PCC = 0.9524 vs required 0.99. All three model configurations tested (MistralForCausalLM + sliding_window=None, MistralForCausalLM + sliding_window=4096, Ministral3ForCausalLM + sliding_window=None) produced PCC in the range 0.9524–0.9540. The gap is consistent with TT Wormhole's BF16 matmul accumulation vs CPU FP32 accumulation accumulated across 34 layers with Q4_K_M dequantized weights. This is a cross-cutting precision issue in the TT-MLIR compiler stack — not a loader bug.

## Fix
**Loader fixes** (`tt-xla/third_party/tt_forge_models/ministral_8b_gguf/causal_lm/pytorch/loader.py`):
- Added `GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")` and `GGUF_TO_TRANSFORMERS_MAPPING["config"]["mistral3"]` key mapping
- Added `GGUF_TO_FAST_CONVERTERS["mistral3"] = GGUFLlamaConverter` and `GGUF_TO_FAST_CONVERTERS["mistral"] = GGUFLlamaConverter`
- In `_patched_load_gguf_checkpoint`: remap `config["model_type"] = "mistral"` and set `config["sliding_window"] = None`
- Patch all four `load_gguf_checkpoint` binding sites: `gguf_utils`, `configuration_utils`, `tokenization_utils_tokenizers`, `tokenization_auto`
- In `_patched_get_gguf_hf_weights_map`: remap `effective_type == "mistral"` → `model_type = "mistral3"` for gguf-py arch lookup

**Tier A fix** (`tt-xla/python_package/tt_torch/backend/passes.py` + `backend.py`):
- Added `clamp_out_of_range_slice_starts(gm)` FX pass that pre-clamps negative `start` arguments in `aten.slice.Tensor` nodes to `-dim_size` when `start < -dim_size`. Uses `node.meta["val"].shape` (FakeTensor from `torch.export.export`) to determine the actual dimension size at compile time.
- Wired into `torch_pass_pipeline` after `bypass_assert_tensor_metadata`.

**Proposed Tier B fix (not implemented)**:
The residual PCC gap requires either preserving FP32 accumulation through the entire TTIR/TTNN lowering pipeline for matmul and its decompositions, or moving to a higher-precision accumulation mode in the TT hardware kernels. This is a cross-cutting change touching multiple matmul-related lowering patterns across tt-mlir and tt-metal.

## Tier B justification
`ttmlir-bf16-precision-not-preserved` is cross-cutting: every matmul lowering pattern in tt-mlir produces BF16 accumulation regardless of the input precision, and fixing it would require coordinated changes across multiple lowering patterns in tt-mlir plus potentially tt-metal kernel changes. The PCC gap is ~5% across all configurations, consistent with accumulated BF16 rounding across 34 layers with dequantized Q4_K_M weights.

Applicable Tier B indicator: **cross-cutting** — requires coordinated changes across multiple files and potentially multiple repos (tt-mlir + tt-metal).

## Verification
- pytest exit: FAIL (PCC 0.9524 vs required 0.99)
- Hardware:    n150
- Duration:    ~142s (model load + inference to completion)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/ministral_8b_gguf/causal_lm/pytorch/loader.py` (loader fixes)
- `tt-xla/third_party/tt_forge_models/ministral_8b_gguf/causal_lm/pytorch/requirements.txt` (added gguf>=0.10.0)
- `tt-xla/python_package/tt_torch/backend/passes.py` (added clamp_out_of_range_slice_starts)
- `tt-xla/python_package/tt_torch/backend/backend.py` (import + call of new pass)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8a5a5e43963f18938d73d8d13a395362c40a79a9 |
| tt-forge-models | a55b9a29b8365e036922bbc22fa476d170c2a04a |
