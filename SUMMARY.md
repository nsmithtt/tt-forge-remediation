# Remediation Summary: llama_3_1_8b_instruct_bnb_nf4-causal_lm-pytorch-3.1_8B_Instruct_BNB_NF4-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_1_8b_instruct_bnb_nf4/causal_lm/pytorch-3.1_8B_Instruct_BNB_NF4-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bnb-nf4-params4bit-incompatible-with-xla-device-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/device/chip_helpers/silicon_sysmem_manager.cpp:326: tt::exception

Local reproduction with bitsandbytes installed shows the proximate cause:

E   RuntimeError: Creating a Parameter from an instance of type Params4bit requires that detach() returns an instance of the same type, but return type Tensor was found instead. To use the type as a Parameter, please correct the detach() semantics defined by its __torch_dispatch__() implementation.

The CI machine (ip-172-31-30-232) uses IOMMU instead of hugepages for sysmem
initialization (hugepage NUMA bind fails with "Operation not permitted"), and
the IOMMU path in pin_or_map_iommu() found that the mapped NOC address did not
match the expected PCIe base address — a device-state symptom that masked the
underlying loader error. With bitsandbytes installed and the loader fixed, the
test passes without triggering any device-level issue.

## Root cause
`bitsandbytes` was not listed in `requirements.txt`, so it was absent in CI
environments that do not pre-install it. When `bitsandbytes` is absent,
`AutoModelForCausalLM.from_pretrained` raises `ImportError` immediately.

With `bitsandbytes` installed, the model loads successfully with all linear
layers stored as `bnb.nn.Linear4bit` (Params4bit parameters). When the test
runner calls `model.to(xla_device)`, PyTorch's `Module._apply()` converts each
parameter and tries to wrap it back as `nn.Parameter`. Because
`Params4bit.detach()` returns a plain `torch.Tensor` instead of a `Params4bit`,
PyTorch raises `RuntimeError: Creating a Parameter from an instance of type
Params4bit requires that detach() returns an instance of the same type`. TT
hardware has no bitsandbytes/CUDA dequantization kernels, so the quantized
representation cannot be used on-device.

Additionally, `load_inputs` used `padding="max_length"` (max_length=128), which
causes PCC degradation on TT for short inputs (known pattern in this codebase).

## Fix
**File**: `tt-xla/third_party/tt_forge_models/llama_3_1_8b_instruct_bnb_nf4/causal_lm/pytorch/loader.py`

1. Added `_dequantize_bnb_model()` helper: after `AutoModelForCausalLM.from_pretrained`,
   iterate all `bnb.nn.Linear4bit` modules and replace each with a standard
   `nn.Linear` whose weights are obtained via `bitsandbytes.functional.dequantize_4bit`
   (CPU-capable since bitsandbytes 0.44+). The dequantized weights are cast to
   `bfloat16`. LLaMA 3.1 8B at bf16 is ~16 GB, which fits in the p150b's 24 GB DRAM.

2. Removed `padding="max_length"` / `truncation=True` / `max_length=` kwargs from
   the tokenizer call in `load_inputs` to avoid the known PCC-drop pattern.

**File**: `tt-xla/third_party/tt_forge_models/llama_3_1_8b_instruct_bnb_nf4/causal_lm/pytorch/requirements.txt`

Added `bitsandbytes>=0.46.1`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    149.35s (0:02:29)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llama_3_1_8b_instruct_bnb_nf4/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/llama_3_1_8b_instruct_bnb_nf4/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1e5d962a3d941aa6b117c46398b8f17649c54e6c |
| tt-forge-models | b1de537efa608c69a14ace0769081d3d8fc5e48b |
