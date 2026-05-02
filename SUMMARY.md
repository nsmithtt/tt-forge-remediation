# Remediation Summary: devstral_small_2_awq-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/devstral_small_2_awq/pytorch-single_device-inference]

## Result
XFAIL — dequantized BF16 model (~51 GB) exceeds p150b single-device DRAM (32 GB); hardware capacity ceiling after loader fix

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-24b-bf16-dram-ceiling

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
ValueError: Unrecognized configuration class <class 'transformers.models.mistral3.configuration_mistral3.Mistral3Config'> for this kind of AutoModel: AutoModelForCausalLM.
```

After loader fix:
```
RuntimeError: TT_FATAL @ bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4196976832 B, free: 76413184 B, largest free block: 37030336 B)
```

## Root cause
Two bugs in the loader, then a hardware capacity ceiling:

1. **Wrong model class (loader):** The checkpoint `cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit` has `model_type=mistral3` and architecture `Mistral3ForConditionalGeneration`. `AutoModelForCausalLM` does not have `Mistral3Config` in its mapping (`Ministral3Config` is present, but not `Mistral3Config`). Loading fails with `ValueError: Unrecognized configuration class`.

2. **Missing compressed-tensors dependency + broken layer forward (loader):** The checkpoint uses compressed-tensors `pack-quantized` INT4 format. Without `compressed-tensors` installed, the import fails. When installed, compressed-tensors loads quantized layers as `torch.nn.Linear` but stores weights as `weight_packed` / `weight_scale` / `weight_shape` instead of `weight`, and monkey-patches the instance `forward` method. This makes the layer's standard forward call fail with `'Linear' object has no attribute 'weight'`. All quantized layers must be dequantized to standard `nn.Linear` before TT device inference.

3. **Hardware capacity ceiling:** After fixing the loader, the dequantized BF16 model is ~51 GB (40 text-decoder layers × ~1.1 GB + vision tower + embeddings). The p150b device exposed 8 DRAM banks × ~4 GB = ~32 GB to the TTNN allocator. The model exceeds this capacity. OOM occurs during the tilize step when loading the first large weight tensor onto device.

## Fix
**In `tt_forge_models` (`mistral/devstral_small_2_awq/pytorch/`):**

- `loader.py`: Replace `AutoModelForCausalLM.from_pretrained` with `Mistral3ForConditionalGeneration.from_pretrained`. Add `_dequantize_compressed_tensors_and_restore_linear(model, dtype)` helper that walks all `nn.Linear` modules with `quantization_scheme` attribute, calls `PackedQuantizationCompressor.decompress()` to unpack INT4 weights, and replaces each with a standard `nn.Linear` holding the BF16 weight.

- `requirements.txt`: Add `compressed-tensors` dependency.

**In `tt-xla` (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):**

- Add `mistral/devstral_small_2_awq/pytorch-single_device-inference` with `status: KNOWN_FAILURE_XFAIL` and reason documenting the DRAM ceiling.

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    335.85s
- Tier A attempts: N/A

## Files changed
- `mistral/devstral_small_2_awq/pytorch/loader.py` (tt-forge-models)
- `mistral/devstral_small_2_awq/pytorch/requirements.txt` (tt-forge-models, new)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5b8bdf49877afff94fa59c4632abd08a0f1a2645 |
| tt-forge-models | 2bdb97b72e250d88e0f7be599920eb1ff2ab39de |
