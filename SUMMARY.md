# Remediation Summary: qwen_3/embedding/pytorch-Embedding_8B_NVFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3/embedding/pytorch-Embedding_8B_NVFP4-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
compressed-tensors-nvfp4-unsupported-float-type

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization.
(On CI where compressed-tensors was installed: ValueError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/pjrt_implementation/src/utils/data_type_utils.cc:163: tt::exception — "Unsupported float type" because TT hardware has no FP4 or FP8 data type support.)

## Root cause
Loader bug in the `qwen_3/embedding/pytorch` loader (tt_forge_models). The model `alexliap/Qwen3-Embedding-8B-NVFP4` uses the `nvfp4-pack-quantized` compressed-tensors format: weights are stored as 4-bit floats (FP4) with FP8 (float8_e4m3fn) group scales. Two issues:

1. `compressed-tensors` package was not in requirements.txt — loading fails immediately with ImportError. On CI where it was installed, the quantized (FP4/FP8) tensors were passed to TT hardware which has no FP4 data type, triggering `TT_THROW("Unsupported float type")` at `data_type_utils.cc:163`.

2. No `run_compressed=False` flag was set, so compressed-tensors kept FP4 weights as-is rather than dequantizing to BF16.

Additionally, compressed-tensors 0.15.x attaches instance-level `forward` overrides to each `nn.Linear` module after loading. These overrides conflict with TT-XLA's `__torch_function__` dispatch. The decoder-based Qwen3 model also returns KV cache outputs by default (`use_cache=True`), which causes device timeout and PCC failures on TT hardware.

## Fix
Changes in `tt_forge_models` on branch `remediation/qwen3-embedding-8b-nvfp4-fix`:

1. **`qwen_3/embedding/pytorch/requirements.txt`** (new file): added `compressed-tensors` dependency.

2. **`qwen_3/embedding/pytorch/loader.py`**: For the `QWEN_3_EMBEDDING_8B_NVFP4` variant in `load_model`:
   - Load `AutoConfig` and set `config.quantization_config["run_compressed"] = False` (config is a dict) before calling `AutoModel.from_pretrained`. This causes compressed-tensors to dequantize FP4 weights to BF16 at load time, producing a standard BF16 model the TT compiler can handle.
   - After loading, iterate all modules and delete any instance-level `forward` attribute (`del m.__dict__["forward"]`) to remove compressed-tensors dispatch hooks.
   - Set `model.config.use_cache = False` to suppress KV cache outputs from the decoder backbone and avoid device→host transfer overhead / PCC noise.

The dequantized 8B model is ~16 GB BF16, which fits within the p150b's 24 GB DRAM. On n150 (12 GB), `enable_weight_bfp8_conversion: true` would be needed (as already configured for the standard `Embedding_8B` variant).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    212.88s (0:03:32)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/qwen_3/embedding/pytorch/requirements.txt` (new)
- `tt_forge_models/qwen_3/embedding/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8f4479ef4a76a98775b73edbeec9d062d198a6e2 |
