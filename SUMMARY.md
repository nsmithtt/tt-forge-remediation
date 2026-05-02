# Remediation Summary: mira_v1_28_dpo_i1_gguf-causal_lm-pytorch-Mira_v1.28_DPO_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mira_v1_28_dpo_i1_gguf/causal_lm/pytorch-Mira_v1.28_DPO_i1_GGUF-single_device-inference]

## Result
XFAIL — 27B Gemma3-based model (~54 GB BF16) exceeds single-device p150b DRAM capacity (~32-34 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-dram-oom-27b-gemma3

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
```

Actual terminal failure after loader and slice fixes:
```
TT_FATAL: Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks (allocated: 4225171776 B, free: 48218240 B)
```
(Occurred in `ttnn::tilize` during first inference execution, after ~15 min of model loading and compilation)

## Root cause
Mira v1.28 DPO is a fine-tune of Gemma 3 27B (hidden_size=5376, 62 transformer layers). Dequantized to BF16 the model weights occupy approximately 54 GB, which far exceeds the ~32-34 GB DRAM available on a single blackhole p150b device.

Two loader-level bugs were encountered and fixed along the way (see Fix section), but neither was the terminal failure. After fixing those bugs, the model loaded and compiled successfully (~15 min), then OOMed on the first inference attempt when TTNN tried to tilize the first input tensor with only 48 MB DRAM remaining.

The reported failure (`RuntimeError: Value out of range ... got -1023`) was a secondary issue: Gemma-3's sliding_window=1024 produces `kv[:, :, -(sliding_window-1):, :]` which with seq_len=23 becomes a start index of -1023, rejected by XLA's strict bounds checking. This was a real bug and has been fixed (see Fix section), but was not the root cause of the test failure.

## Fix

### Loader fixes (tt_forge_models, remediation branch)

**Fix 1: Broken patcher chain (`gguf-load-checkpoint-model-to-load-kwarg`)**
Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`. Other loaders in the test session (e.g. `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`) patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a function that only accepts `(gguf_path, return_tensors=False)`, missing `**kwargs`. When mira's loader runs `AutoModelForCausalLM.from_pretrained()`, it hits the patched function and raises `TypeError`.

Fix: Added `_unwrap_to_transformers()` (BFS through `__closure__` and `__globals__["_orig*"]` keys) to locate the true transformers function, and a `_restored_load_gguf_checkpoint()` context manager that temporarily restores the true function around the `from_pretrained` call.

File: `tt_forge_models/mira_v1_28_dpo_i1_gguf/causal_lm/pytorch/loader.py`
Commit: `ffa1e8821d` on remediation branch of tt_forge_models.

**Fix 2: Missing chat_template guard (`gguf-tokenizer-no-chat-template`)**
Added `if self.tokenizer.chat_template is not None:` guard before `apply_chat_template`, falling back to a plain `"User: {text}\nAssistant:"` template.

File: `tt_forge_models/mira_v1_28_dpo_i1_gguf/causal_lm/pytorch/loader.py` (same commit)

### Compiler frontend fix (tt-xla, remediation branch)

**Fix 3: OOB slice start for sliding-window attention (`aten-slice-tensor-out-of-bounds-start`)**
Gemma-3 sliding_window=1024 generates `aten.slice.Tensor(kv, 2, -1023, MAX_INT)` when seq_len=23. XLA validates slice bounds strictly and rejects start=-1023 (valid range [-23, 22]).

Fix: Added slice-bounds clamping block in `TorchFunctionOverride.__torch_function__` to clamp start/end to `[-size, size]` before dispatch.

File: `tt-xla/python_package/tt_torch/torch_overrides.py:34-48`
Commit: `f91d893b6` on remediation branch of tt-xla.

### Hardware-class XFAIL (tt-xla test config)

Marked `KNOWN_FAILURE_XFAIL` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.
Commit: `777d3cafa` on remediation branch of tt-xla.

## Verification
- pytest exit: FAIL (OOM after all fixes applied)
- Hardware:    blackhole-p150b
- Duration:    985.57s (0:16:25) — model loaded and compiled, OOMed on first execution
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mira_v1_28_dpo_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 777d3cafaed2406a1142b26b89ebdcddc8b1afdf |
| tt-forge-models | ffa1e8821d2554c40d32d17beeefb2fe3b6e8fd8 |
