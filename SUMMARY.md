# Remediation Summary: gpt_oss_20b_heretic_gguf-causal_lm-pytorch-20B_heretic_uncensored_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_heretic_gguf/causal_lm/pytorch-20B_heretic_uncensored_GGUF-single_device-inference]

## Result
XFAIL — Hardware capacity: 20B Qwen3MoE BF16 ~40 GB exceeds all single-device DRAM (n150 12 GB, p150b 32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-shard-spec-wrong-mlp-attr

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'

Raised in `load_shard_spec` when `_safely_put_workload_on_device` calls it during
single-device inference setup. The method assumed LLaMA-style MLP with `up_proj`,
`gate_proj`, and `down_proj`, but `mradermacher/OpenAI-gpt-oss-20B-...` is a
Qwen3 MoE model whose layers use `Qwen3MoeSparseMoeBlock` (no `up_proj`).

## Root cause
Two issues:

1. **Loader bug (AttributeError)**: The `load_shard_spec` method in the loader
   accesses `layer.mlp.up_proj` on every layer. The GGUF file stores the model
   with `general.architecture = qwen3moe`, so transformers loads it as
   `Qwen3MoeForCausalLM`. MoE layers are `Qwen3MoeSparseMoeBlock`, which has no
   `up_proj` attribute. `_safely_put_workload_on_device` calls `load_shard_spec`
   unconditionally, causing `AttributeError`.

2. **Hardware capacity (XFAIL)**: After dequantization by transformers GGUF loading,
   the 20B Qwen3 MoE model is stored at BF16 precision. 20B params × 2 bytes/param
   ≈ 40 GB, which exceeds all single-device DRAM (n150: 12 GB, p150b: 32 GB).

## Fix
**Loader fix** (`tt_forge_models` `gpt_oss_20b_heretic_gguf/causal_lm/pytorch/loader.py`):
- Removed the broken `load_shard_spec` method (assumed LLaMA MLP, model is Qwen3 MoE)
- Removed the `get_mesh_config` method (not needed for single-device inference)

**Test config** (`tt-xla` `tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `KNOWN_FAILURE_XFAIL` entry for this test with explanation of hardware capacity ceiling.

## Verification
- pytest exit: not-run (hardware-class XFAIL before silicon run)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `gpt_oss_20b_heretic_gguf/causal_lm/pytorch/loader.py` (tt-forge-models): removed `load_shard_spec` and `get_mesh_config`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla): added KNOWN_FAILURE_XFAIL

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d9d38bd1e63c3fa95148e765dc12bd364f0f8984 |
| tt-forge-models | 51a81e879919b0ac9b7ea1a04c01c0060ef17dad |
