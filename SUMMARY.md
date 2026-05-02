# Remediation Summary: mlx_community_qwen3_coder_next_6bit-causal_lm-pytorch-Coder_Next_6bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_qwen3_coder_next_6bit/causal_lm/pytorch-Coder_Next_6bit-single_device-inference]

## Result
XFAIL — Qwen3-Coder-Next-6bit (79.7B parameters, ~159 GB BF16) exceeds single-device DRAM on p150b (~34 GB); hardware capacity ceiling

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
mlx-affine-quant-no-quant-method

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   ValueError: The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized
```
(The originally reported `MHLO -> StableHLO conversion failed` was from an older transformers version that did not validate `quant_method` before downloading weights.)

## Root cause
**Loader layer.** `mlx-community/Qwen3-Coder-Next-6bit` stores quantization metadata in `config.json` as `{"group_size": 64, "bits": 6, "mode": "affine"}` with no `quant_method` field. `transformers 5.x` validates `quant_method` when `quantization_config` is present and raises `ValueError` before any weights are loaded.

After stripping the metadata the model can be instantiated via `from_config`, but the actual weights are MLX affine 6-bit packed (uint32 LSB-first bit stream, per-group bfloat16 scales and biases). The checkpoint has per-layer 8-bit overrides for `mlp.gate` and `mlp.shared_expert_gate`. Expert weights are stored as two separate 3D tensors (`switch_mlp.gate_proj` + `switch_mlp.up_proj`) that must be concatenated along dim=1 to form the single `experts.gate_up_proj` tensor expected by `Qwen3NextForCausalLM`.

Independently of the loader bug, the model has 79.7B parameters which requires ~159 GB BF16 on a single device. The p150b provides approximately 34 GB of device DRAM, making single-device inference impossible. This is a hardware capacity ceiling.

## Fix
**Loader fix** (`tt_forge_models/mlx_community_qwen3_coder_next_6bit/causal_lm/pytorch/loader.py`):
1. Extract quantization parameters from config before deleting `quantization_config` and `quantization` attributes.
2. Build model from config with `AutoModelForCausalLM.from_config()`.
3. Implement `_unpack_6bit()` using numpy for cross-word boundary handling in the LSB-first 6-bit packed bit stream.
4. Implement `_unpack_8bit()` for the per-layer 8-bit override layers.
5. Implement `_dequantize()` that handles both 2D (standard linear) and 3D (batched expert) weight shapes with per-group scale+bias expansion.
6. Implement `_process_shard()` that dequantizes one safetensors shard at a time, collecting pending gate projections until the corresponding up projection arrives, then concatenates `switch_mlp.gate_proj` + `switch_mlp.up_proj` → `experts.gate_up_proj`.
7. Call `model.load_state_dict(strict=False)` + `model.tie_weights()`.

**Test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
Added `KNOWN_FAILURE_XFAIL` entry with hardware capacity reason.

## Verification
- pytest exit: FAIL (not run on silicon — hardware capacity ceiling)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mlx_community_qwen3_coder_next_6bit/causal_lm/pytorch/loader.py` — full rewrite with 6-bit MLX dequantization
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry added

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4d1fd101ef14e1ec783435821c3844553b5f898e |
| tt-forge-models | 3838ff6bdc4b633118d0d1a7c3903aa543b0371f |
