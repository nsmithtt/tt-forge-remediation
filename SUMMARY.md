# Remediation Summary: dicta_lm_3_0_24b_thinking_i1_gguf-causal_lm-pytorch-3.0_24B_Thinking_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dicta_lm_3_0_24b_thinking_i1_gguf/causal_lm/pytorch-3.0_24B_Thinking_i1_GGUF-single_device-inference]

## Result
XFAIL — 24B Q4_K_M model (~13 GB) exceeds single-device DRAM (~4 GB on n150/wormhole)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-24b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4196977728 B, free: 76412288 B, largest free block: 37030336 B)
```

## Root cause
The original failure (`ImportError: Please install torch and gguf>=0.10.0`) was a missing
requirement. Two additional loader-layer bugs were encountered and fixed along the way:

1. **gguf requirement missing** (`dicta_lm_3_0_24b_thinking_i1_gguf`): `gguf>=0.10.0` was not
   listed in requirements.txt, causing ImportError when the GGUF loader was first used.

2. **`get_gguf_hf_weights_map` missing `processor` arg**: Two loaders (`qwen_3_vl_8b_thinking_heretic_i1_gguf`
   and `qwen_3_5_claude_distilled_gguf`) defined `patched_get_gguf_hf_weights_map` without the
   `processor` positional parameter that transformers 5.x added to `get_gguf_hf_weights_map`. During
   test discovery these loaders are imported and globally patch the function. When the dicta_lm test
   ran, the chained patches received `processor` as a positional arg that the qwen3vl patch mapped to
   `model_type`, then the following call also passed `model_type` as a keyword, causing
   `TypeError: got multiple values for argument 'model_type'`.

After these loader fixes the model loaded successfully but immediately OOMed on device. The
DictaLM-3.0-24B-Thinking-i1-GGUF uses Q4_K_M quantization which yields approximately 13 GB of
weights. A single n150 wormhole device has approximately 4 GB of DRAM. The runtime error confirms:
`bank_size = 4273390016 B (≈4 GB), allocated = 4196977728 B (≈4 GB), free = 76412288 B (≈73 MB)`.
This is a genuine hardware capacity ceiling.

## Fix
Two loader fixes were applied in `tt_forge_models` on branch
`remediation/dicta_lm_3_0_24b_thinking_i1_gguf-causal_lm-pytorch-3.0_24B_Thinking_i1_GGUF-single_device-inference`:

1. `dicta_lm_3_0_24b_thinking_i1_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`
2. `qwen_3_vl_8b_thinking_heretic_i1_gguf/image_to_text/pytorch/loader.py` — added `processor=None`
   to `patched_get_gguf_hf_weights_map` and passed it to `orig_weights_map`
3. `qwen_3_5_claude_distilled_gguf/causal_lm/pytorch/loader.py` — same `processor=None` fix

The test was marked `KNOWN_FAILURE_XFAIL` in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla) because the
underlying cause is hardware capacity, not a compiler bug.

## Verification
- pytest exit: FAIL (OOM at runtime after loader fixes)
- Hardware:    n150
- Duration:    794.89s (0:13:14) for OOM run
- Tier A attempts: N/A

## Files changed
**tt_forge_models** (`remediation/dicta_lm_3_0_24b_thinking_i1_gguf-causal_lm-pytorch-3.0_24B_Thinking_i1_GGUF-single_device-inference`):
- `dicta_lm_3_0_24b_thinking_i1_gguf/causal_lm/pytorch/requirements.txt`
- `qwen_3_vl_8b_thinking_heretic_i1_gguf/image_to_text/pytorch/loader.py`
- `qwen_3_5_claude_distilled_gguf/causal_lm/pytorch/loader.py`

**tt-xla** (`remediation/dicta_lm_3_0_24b_thinking_i1_gguf-causal_lm-pytorch-3.0_24B_Thinking_i1_GGUF-single_device-inference`):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ce7779729 |
| tt-forge-models | a62fb8f8d4 |
