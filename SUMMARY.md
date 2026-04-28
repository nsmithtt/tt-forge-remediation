# Remediation Summary: intern_vl_chat_v1_2-pytorch-V1_2-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[intern_vl_chat_v1_2/pytorch-V1_2-single_device-inference]

## Result
XFAIL — model is ~41B parameters (~83 GB BF16), far exceeding single-device DRAM (n150: 12 GB, p150b: 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-41b-vlm-exceeds-single-device-dram

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
InternVL-Chat-V1-2 uses InternViT-6B (45 layers, hidden=3200) as vision encoder
and a 60-layer LLaMA-based language model (hidden=7168, intermediate=20480), giving
~41B total parameters. At BF16 that is ~83 GB, which far exceeds the DRAM capacity
of any single Tenstorrent device (n150: 12 GB, p150b: 24 GB). The CI failure
`INTERNAL: Error code: 13` is the OOM indicator from the TT runtime.

A secondary loader bug was found during reproduction: transformers 5.x unconditionally
initialises models on `torch.device("meta")`. InternVisionEncoder.__init__ calls
`torch.linspace(0, drop_path_rate, num_hidden_layers).item()` during this meta-device
init pass, causing `RuntimeError: Tensor.item() cannot be called on meta tensors`.
This was triggered by a freshly downloaded version of modeling_intern_vit.py from
the OpenGVLab/InternVL-Chat-V1-2 HuggingFace repo. The loader was fixed to patch
`torch.Tensor.item` to return `0.0` for meta tensors during `from_pretrained`; the
actual values are overwritten when the checkpoint is loaded.

## Fix
**Loader fix** (tt_forge_models):
- `intern_vl_chat_v1_2/pytorch/loader.py`: Wrap `AutoModel.from_pretrained` in a
  context that patches `torch.Tensor.item` to return `0.0` for meta tensors.

**Test config** (tt-xla):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added
  `intern_vl_chat_v1_2/pytorch-V1_2-single_device-inference: KNOWN_FAILURE_XFAIL`.

## Verification
- pytest exit: XFAIL (1 xfailed, 7 warnings in 21.27s)
- Hardware:    n150
- Duration:    21.27s (xfail short-circuit, no silicon execution)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/intern_vl_chat_v1_2/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 02c9328c880a09e00b058c4b307e5fb0bda5f52e |
| tt-forge-models | 1958b0dfb4131d94c7ef501e56aed302dbf323fa |
