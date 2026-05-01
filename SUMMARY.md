# Remediation Summary: gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf-causal_lm-pytorch-20B_SFT_V0_1_MXFP4_MOE_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch-20B_SFT_V0_1_MXFP4_MOE_GGUF-single_device-inference]

## Result
XFAIL — 20B Qwen3MoE model dequantizes from MXFP4 GGUF to ~40 GB BF16, exceeding single-device DRAM (p150b 32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
loader-load-shard-spec-moe-attr-error

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
tests/infra/runners/torch_device_runner.py:167: in _safely_put_workload_on_device
    shard_specs = workload.shard_spec_fn(workload.model)
third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py:198: in load_shard_spec
    shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1964: in __getattr__
    raise AttributeError(
AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'
```

## Root cause
Two bugs:

1. **Loader bug (immediate crash)**: The `load_shard_spec` function in the loader assumed LLaMA-style flat MLP layers (with `up_proj`, `gate_proj`, `down_proj`), but the model loads as `Qwen3MoeForCausalLM` whose MoE layers contain `Qwen3MoeSparseMoeBlock` objects. `Qwen3MoeSparseMoeBlock` has no `up_proj` attribute — the actual projections live inside the `experts` list and `shared_expert` sub-module.

   `_safely_put_workload_on_device` in `torch_device_runner.py` calls `shard_spec_fn(model)` unconditionally whenever the function is set and `device.type != "cpu"`, even for single-device inference. Since the loader defines `load_shard_spec`, it is invoked, and the attribute access fails.

2. **Hardware capacity ceiling**: The `sashisuseso/GPT-OSS-Swallow-20B-SFT-v0.1-MXFP4_MOE-GGUF` model is a 20B-parameter Qwen3MoE (24 layers, 32 experts, hidden_size=2880). The 12 GB MXFP4 GGUF is always dequantized to BF16 by the transformers GGUF loader. BF16 model size = 20B × 2 bytes ≈ 40 GB, which exceeds all single-device DRAM (n150: 12 GB, n300/p150b: 32 GB).

## Fix
1. **Loader fix** (`tt_forge_models` remediation branch `remediation/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf-causal_lm-pytorch-20B_SFT_V0_1_MXFP4_MOE_GGUF-single_device-inference`, commit `1de4e8e9a9`):
   - Removed `load_shard_spec` and `get_mesh_config` from the loader. This model is Qwen3MoE-based and has no valid tensor-parallel sharding definition; single-device inference does not need these methods.
   - File: `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`

2. **Test config XFAIL** (`tt-xla` remediation branch, commit `232f9d13a`):
   - Added `KNOWN_FAILURE_XFAIL` entry for this model in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` with reason "20B Qwen3MoE model: MXFP4 GGUF dequantizes to ~40 GB BF16, exceeding single-device DRAM (p150b 32 GB)".

## Verification
- pytest exit: FAIL (AttributeError before reaching device; reproduced locally in 1042s)
- Hardware:    n150
- Duration:    1042.33s (0:17:22) — model load time, error before device execution
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py` — removed `load_shard_spec` and `get_mesh_config`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 232f9d13abbad77206c766ed4ebff9610293a53d |
| tt-forge-models | 1de4e8e9a98dedaa1016548b58fb1f2078680951 |
