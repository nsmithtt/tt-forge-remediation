# Remediation Summary: exaone_3_5_awq-causal_lm-pytorch-3.5_7.8B_Instruct_AWQ-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[exaone_3_5_awq/causal_lm/pytorch-3.5_7.8B_Instruct_AWQ-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-check-model-inputs-missing-and-awq-qlinear-not-implemented-on-xla

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ImportError: cannot import name 'check_model_inputs' from 'transformers.utils.generic' (/home/nsmith/hf-bringup/tt-xla/.local_venv/lib/python3.12/site-packages/transformers/utils/generic.py)

## Root cause
Two loader-layer bugs:

**Bug 1**: The cached `modeling_exaone.py` (commit `e50260813863d02f6aafb889bb2820d447704c6c` from HuggingFace) imports `check_model_inputs` and `maybe_autocast` from `transformers.utils.generic`. `maybe_autocast` exists in transformers 5.2.0, but `check_model_inputs` was only added between 5.3.0 and 5.7.0. In transformers 5.7.0 it is already deprecated as an alias for `merge_with_config_defaults`, which does exist in 5.2.0.

**Bug 2**: After fixing the import, `gptqmodel`'s `TorchAtenAwqLinear` (the backend for AWQ quantized linear layers) sets `linear_mode = "inference"` during the first CPU forward pass and calls `transform_cpu()` to pack weights for `torch.ops.aten._weight_int4pack_mm_for_cpu`. On the subsequent TT XLA forward pass, `_fused_op_forward` is invoked with an XLA tensor (`x.device.type == "xla"`), which raises `NotImplementedError` because only CPU tensors are supported. This blocks TT XLA compilation entirely.

## Fix
Two fixes in `exaone_3_5_awq/causal_lm/pytorch/loader.py` in `tt_forge_models`:

1. **check_model_inputs shim**: Before `from_pretrained` is called, inject `check_model_inputs` into `transformers.utils.generic` if absent, pointing it to `merge_with_config_defaults` (its equivalent in 5.2.0).

2. **AWQ dequantization**: After `from_pretrained` but before any forward pass, replace all `TorchAtenAwqLinear` modules with plain `nn.Linear` modules using weights dequantized via `awq_weight_dequantize`. The dequantization must happen before any forward pass so that `transform_cpu()` has not yet overwritten the packed buffers. This produces a device-agnostic float model that TT XLA can compile normally.

Also added `gptqmodel` to `exaone_3_5_awq/causal_lm/pytorch/requirements.txt`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    199.36s (0:03:19)
- Tier A attempts: N/A

## Files changed
- `exaone_3_5_awq/causal_lm/pytorch/loader.py` (tt_forge_models)
- `exaone_3_5_awq/causal_lm/pytorch/requirements.txt` (tt_forge_models, new file)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 41ed7d6f49bad18c4b8ba1a91c1c30faa591b09e |
| tt-forge-models | 837074f5435c652633ae7f862611272dc539abb4 |
