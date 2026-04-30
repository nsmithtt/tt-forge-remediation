# Remediation Summary: devstral_small_2_24b_instruct_abliterated_i1_gguf-causal_lm-pytorch-Devstral_Small_2_24B_Instruct_Abliterated_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[devstral_small_2_24b_instruct_abliterated_i1_gguf/causal_lm/pytorch-Devstral_Small_2_24B_Instruct_Abliterated_i1_GGUF-single_device-inference]

## Result
XFAIL — 24B bfloat16 model (~48 GB) exceeds single-device DRAM on all current TT hardware

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (loader layer):
```
ValueError: GGUF model with architecture mistral3 is not supported yet.
```
Intermediate failure after loader fix (tt-xla layer):
```
RuntimeError: Value out of range (expected to be in range of [-9, 8], but got -4095)
While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807))
```
Final failure (hardware capacity):
```
RuntimeError: TT_FATAL @ .../tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4196980928 B, free: 76409088 B, largest free block: 37026240 B)
```

## Root cause
Three separate issues found and fixed, with the final disposition being hardware capacity:

**Loader (fixed):** The GGUF file uses architecture tag `mistral3` (Mistral Small 3.x text
backbone). Transformers 5.x does not recognise `mistral3` in `GGUF_SUPPORTED_ARCHITECTURES`,
`GGUF_TO_TRANSFORMERS_MAPPING`, or `GGUF_TO_FAST_CONVERTERS`. The loader also needed to
rewrite `model_type: mistral3 → mistral` so `AutoModelForCausalLM` selects `MistralForCausalLM`
instead of the multimodal `Mistral3ForConditionalGeneration`. Additionally, `gguf-py`'s
`MODEL_ARCH_NAMES` only maps `mistral3` (not `mistral`) to the weight tensor name table,
so `get_gguf_hf_weights_map` needed a reverse-translation wrapper. Multiple other GGUF loaders
patch `load_gguf_checkpoint` at module level with the old 2-arg signature, causing ordering
races; the fix uses context managers to install/restore patches around each `from_pretrained`
call and walks the `__globals__` chain to find the real transformers function.

**tt-xla (fixed):** Mistral uses a sliding-window KV cache with `sliding_window=4096`. During
FX graph tracing the cache slice `full_value_states[:, :, -self.sliding_window + 1:, :]`
becomes `aten.slice(tensor, dim=2, start=-4095, end=MAX_INT)`. When the traced input has only
9 tokens, the XLA lazy backend validates `start >= -size` and raises `Value out of range`.
PyTorch eager silently clamps out-of-range negative indices; the fix adds that clamping in
`TorchFunctionOverride.__torch_function__` in `tt_torch/torch_overrides.py`.

**Hardware capacity (XFAIL):** After both fixes, the model loads and compiles successfully,
but the device OOMs when trying to tilize an input tensor. The 24B parameter model, when
dequantized by transformers from GGUF Q4_K_M to bfloat16, is ~48 GB. The Blackhole p150b
device on this machine has ~32 GB DRAM (8 banks × ~4 GB). Peak allocations fill ~3.91 GB
per bank, leaving only ~73 MB free — not enough for the 320 MB tilize buffer. This is
not a compiler bug; it is a genuine hardware capacity ceiling.

## Fix
1. **Loader** (`tt_forge_models`, remediation branch):
   - `devstral_small_2_24b_instruct_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
   - Register `mistral3` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`,
     and `GGUF_TO_FAST_CONVERTERS` at import time (permanent dict mutations)
   - Context-manager patch that installs `_mistral3_load_gguf_checkpoint` (rewrites
     `model_type: mistral3 → mistral`) and `_mistral3_get_gguf_hf_weights_map` (translates
     back to `mistral3` for gguf-py arch lookup) around each `from_pretrained` call
   - `_find_real_gguf_fn()` walks `__globals__` and `__closure__` to unwrap other loaders'
     module-level `load_gguf_checkpoint` patches and reach the real transformers function

2. **tt-xla** (`tt-xla`, remediation branch):
   - `python_package/tt_torch/torch_overrides.py`
   - Added `aten.slice.Tensor` interception in `TorchFunctionOverride.__torch_function__`:
     when `start < -size` or `end < -size`, clamp to `-size` before dispatch (matches PyTorch
     eager semantics)
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
   - Added `KNOWN_FAILURE_XFAIL` entry for this test with reason explaining OOM

## Verification
- pytest exit: FAIL (hardware OOM)
- Hardware:    blackhole-p150b
- Duration:    665.97s (0:11:05)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/devstral_small_2_24b_instruct_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aea0042902feedc58d65bbaf8eb51e67aa638a5e |
| tt-forge-models | 0562143ca5319fc9833d3f02ceeda28e9bcc01e6 |
