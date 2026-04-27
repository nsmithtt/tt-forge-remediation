# Remediation Summary: crow_9b_opus_4_6_distill_heretic_qwen3_5/conditional_generation/pytorch-9B_Opus_4.6_Distill_Heretic_Qwen3.5-single_device-inference

## Skill version
13

## Test
tests/runner/test_models.py::test_all_models_torch[crow_9b_opus_4_6_distill_heretic_qwen3_5/conditional_generation/pytorch-9B_Opus_4.6_Distill_Heretic_Qwen3.5-single_device-inference]

## Result
FAIL — Qwen3.5 vision encoder calls `grid_thw.tolist()` inside `fast_pos_embed_interpolate`; the TT PJRT backend cannot transfer this device tensor to host, failing with `Bad StatusOr access: INTERNAL: Error code: 13`.

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Full traceback chain:
```
transformers/models/qwen3_5/modeling_qwen3_5.py:1938: in forward
    outputs = self.model(...)
transformers/models/qwen3_5/modeling_qwen3_5.py:1663: in forward
    image_outputs = self.get_image_features(...)
transformers/models/qwen3_5/modeling_qwen3_5.py:1546: in get_image_features
    vision_output = self.visual(...)
transformers/models/qwen3_5/modeling_qwen3_5.py:1239: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
transformers/models/qwen3_5/modeling_qwen3_5.py:1162: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
**Layer: compiler frontend (tt-xla) / runtime (tt-metal)**

The `Qwen3_5ForConditionalGeneration` vision encoder (`modeling_qwen3_5.py:1162`) calls `grid_thw.tolist()` inside `fast_pos_embed_interpolate` to obtain the image grid dimensions as Python integers. The `image_grid_thw` tensor is part of the model inputs and is placed on the TT device by the test framework. Calling `.tolist()` on a TT device tensor requires a synchronous device-to-host data transfer. The TT PJRT runtime does not support this transfer path and returns `INTERNAL: Error code: 13`.

This is the same class of failure as "int → CPU transfer unsupported" in the skill's compiler-stack bug list. The `grid_thw` tensor contains small integer metadata (grid dimensions), not bulk compute data, but the backend cannot transfer it regardless.

## Fix
**Two loader-layer bugs were fixed (both required to reach the compiler-stack failure):**

1. **`apply_chat_template` image-token truncation** (`loader.py`): The original code called `apply_chat_template(tokenize=True)`, which internally expanded the image placeholder to 2752 `<|image_pad|>` tokens in the text, then the tokenizer's `init_kwargs` (`max_length=2048`, `truncation_strategy='longest_first'`) truncated `input_ids` to 2044. The text/ids mismatch raised `ValueError`. Fixed by switching to `tokenize=False` and calling the processor separately.

2. **Processor call missing `truncation=False`** (`loader.py`): Even after the two-stage fix, the tokenizer's `init_kwargs` still caused truncation when calling `self.processor(...)`. Fixed by explicitly passing `truncation=False` to override the tokenizer's stored init_kwargs.

**Compiler-stack fix (proposed, not implemented):**

The TT PJRT backend needs to support device-to-host tensor data reads during model execution (`.tolist()`, `.item()`). Specifically, `python_package/tt_torch/torch_overrides.py` intercepts all `__torch_function__` calls but passes `.tolist()` through unchanged to the PJRT client, which fails. A fix would require either:
- Implementing a device→host tensor transfer fallback in the PJRT client for `.tolist()` / `.item()` calls
- Or treating `image_grid_thw` as a host-side constant in the compiled graph (marking it as non-device metadata)

The tt-xla layer is the right place for this fix since `torch_overrides.py` already intercepts `__torch_function__` and could detect `.tolist()` calls on device tensors and route them through a CPU copy.

## Verification
FAIL — compiler-stack bug reached; no silicon run attempted.

## Files changed
- `tt-xla/third_party/tt_forge_models/crow_9b_opus_4_6_distill_heretic_qwen3_5/conditional_generation/pytorch/loader.py`
  - Switched `apply_chat_template` to `tokenize=False` + two-stage processor call
  - Added `truncation=False` to processor call to prevent tokenizer `init_kwargs` truncation

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c3eb987c55c5d50ae449229013f38b9d19d85444 |
| tt-forge-models | af993a68da25805dca60763a4e8ce590fac43049 |
