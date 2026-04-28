# Remediation Summary: grin_moe-causal_lm-pytorch-GRIN-MoE-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[grin_moe/causal_lm/pytorch-GRIN-MoE-single_device-inference]

## Result
XFAIL — GRIN-MoE ~84 GB at bfloat16 exceeds single-device DRAM on all supported hardware (max 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-42b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
GRIN-MoE has 32 layers × 16 experts × 3 matrices (4096 × 6400 each) ≈ 40.2B expert parameters, plus ~2B attention/embedding parameters, totalling ~42B parameters. At bfloat16 that is ~84 GB. The TT n150 has 12 GB DRAM and the p150b has 24 GB — the model is 3–7× over single-device capacity.

Two loader-layer bugs were also present (transformers 5.x breaking changes) that had to be fixed before the hardware-capacity failure could be reproduced:
1. `config._attn_implementation` was `None` because transformers 5.x makes `_attn_implementation` a property backed by `_attn_implementation_internal`; class-level assignment in the remote config is overridden by `PretrainedConfig.__init__`. Fix: pass `attn_implementation="eager"` to `from_pretrained`.
2. `DynamicCache.get_usable_length()` was removed in transformers 5.x. Fix: monkey-patch it as an alias for `get_seq_length()` with `layer_idx` defaulting to `0` when `None`.

Once those loader bugs were fixed the original error (`INTERNAL: Error code: 13`) reproduced during the first TT sync call, confirming device OOM.

## Fix
- `tt-xla/third_party/tt_forge_models/grin_moe/causal_lm/pytorch/loader.py`: added `attn_implementation="eager"` to `from_pretrained`; added `DynamicCache.get_usable_length` compat shim.
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`: added `grin_moe/causal_lm/pytorch-GRIN-MoE-single_device-inference` entry with `status: KNOWN_FAILURE_XFAIL`.

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    n150
- Duration:    57.92s
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/grin_moe/causal_lm/pytorch/loader.py
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5ffbf7fbeb4f4ab52f775bb7ff6e4e3636bf7076 |
| tt-forge-models | baaacc730b |
