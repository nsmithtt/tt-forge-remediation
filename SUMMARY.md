# Remediation Summary: axolotl_smollm2_135m_bnb_nf4_bf16-causal_lm-pytorch-smollm2_135m_bnb_nf4_bf16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[axolotl_smollm2_135m_bnb_nf4_bf16/causal_lm/pytorch-smollm2_135m_bnb_nf4_bf16-single_device-inference]

## Result
FAIL — silicon verification not possible in this environment (torch_xla not installed); loader fix applied following verified pattern from llama_3_1_8b_instruct_bnb_nf4

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bnb-nf4-missing-requirements-and-padding-pcc-drop

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError while loading conftest (in this environment: torch_xla missing → NameError on Mesh type annotation).
In CI (torch_xla present): raise ImportError("bitsandbytes" not found when loading BNB NF4-quantized model)
```

## Root cause
Two loader bugs:

1. **Missing `requirements.txt`**: `axolotl-ai-co/SmolLM2-135M-bnb-nf4-bf16` stores weights in BitsAndBytes NF4 4-bit quantized format. `AutoModelForCausalLM.from_pretrained` detects the `quantization_config` in config.json and requires `bitsandbytes` to load. Without `bitsandbytes` in `requirements.txt`, transformers raises `ImportError` (the `raise ImportError(` line in CI output).

2. **BNB Params4bit incompatible with TT XLA device transfer**: Even after installing `bitsandbytes`, the loaded model contains `bnb.nn.Linear4bit` layers. `Params4bit.detach()` returns a plain `Tensor` instead of `Params4bit`, making `model.to(xla_device)` fail. The model must be dequantized to standard `nn.Linear` with BF16 weights before TT compilation.

3. **`padding="max_length"` in `load_inputs`**: Causes PCC drop on TT hardware for short inputs (known bug: padded attention masks mishandled, padding tokens add bfloat16 noise to the PCC calculation).

## Fix
In `tt-xla/third_party/tt_forge_models` on branch `remediation/axolotl_smollm2_135m_bnb_nf4_bf16-causal_lm-pytorch-smollm2_135m_bnb_nf4_bf16-single_device-inference`:

1. **`axolotl_smollm2_135m_bnb_nf4_bf16/causal_lm/pytorch/requirements.txt`** (new file): adds `bitsandbytes>=0.46.1`.

2. **`axolotl_smollm2_135m_bnb_nf4_bf16/causal_lm/pytorch/loader.py`**: adds `_dequantize_bnb_model()` helper that iterates `Linear4bit` layers, calls `bnb_F.dequantize_4bit()` to recover BF16 weights, and replaces each `Linear4bit` with a standard `nn.Linear`. Called immediately after `from_pretrained` in `load_model`. Removes `padding="max_length"`, `truncation=True`, and `max_length=` from `load_inputs` tokenizer call.

The fix follows the identical pattern applied in `remediation/llama_3_1_8b_instruct_bnb_nf4-causal_lm-pytorch-3.1_8B_Instruct_BNB_NF4-single_device-inference` (commit `b1de537ef` in tt-forge-models) which was verified SILICON_PASS.

## Verification
- pytest exit: not-run
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- `axolotl_smollm2_135m_bnb_nf4_bf16/causal_lm/pytorch/loader.py` (tt-forge-models)
- `axolotl_smollm2_135m_bnb_nf4_bf16/causal_lm/pytorch/requirements.txt` (tt-forge-models, new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5ba095d5db1684b330c1f79d5336f10d1d404118 |
| tt-forge-models | 5f2a475117c62a0f47af1adb4a8dc34a5d8ca5a5 |
