# Remediation Summary: fbopt_350m_8bit-causal_lm-pytorch-350m-8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fbopt_350m_8bit/causal_lm/pytorch-350m-8bit-single_device-inference]

## Result
SILICON_PASS — loader dequantizes bitsandbytes int8 weights manually for TT hardware

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bnb-int8-quantized-checkpoint-requires-cuda

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: Using `bitsandbytes` 8-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

The error occurs in `transformers/quantizers/quantizer_bnb_8bit.py:60` when `from_pretrained` detects `quantization_config.quant_method == "bitsandbytes"` in the model's `config.json` and requires bitsandbytes to be installed.

## Root cause
The checkpoint `yec019/fbopt-350m-8bit` was saved with bitsandbytes LLM.int8() quantization. The `config.json` contains `{"quant_method": "bitsandbytes", "load_in_8bit": true, ...}`, so transformers' `from_pretrained` path requires bitsandbytes. The `model.safetensors` stores all linear-layer weights as `torch.int8` with per-row scale tensors named `*.SCB` (146 int8 weight tensors, 146 SCB scale tensors). Bitsandbytes is CUDA-only and not installed in the TT venv, making the model unloadable via the standard path.

The fix lives in the loader layer: bitsandbytes quantization is a checkpoint-format concern, not a compiler-stack concern.

## Fix
`tt_forge_models/fbopt_350m_8bit/causal_lm/pytorch/loader.py`:

1. Detect the bitsandbytes quantization config in the loaded `AutoConfig` (checking `quantization_config` as a dict since transformers 5.x returns a raw dict, not an object).
2. Strip `config.quantization_config = None` so `from_config` builds a standard unquantized `OPTForCausalLM`.
3. Download `model.safetensors` via `hf_hub_download`, then manually dequantize all int8 weight matrices: `W_fp = W_int8.float() * (SCB.view(-1, 1) / 127.0)`.
4. Load the dequantized state dict via `model.load_state_dict(state_dict, strict=False)` (strict=False because `lm_head.weight` is tied to `embed_tokens.weight` and absent from the file).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    88.57s (0:01:28)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/fbopt_350m_8bit/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | f3edad8caaec35b4b8520d7e7d882ced989f2b30 |
