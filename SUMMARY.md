# Remediation Summary: param2_17b_a2_4b_thinking-causal_lm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[param2_17b_a2_4b_thinking/causal_lm/pytorch-param2_17b_a2_4b_thinking-single_device-inference]

## Result
XFAIL — 17B BF16 model (~34 GB) exceeds single-device DRAM capacity (~32 GB on p150b)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
param2moe-transformers5x-loader-bugs, hardware-class-dram-oom-17b-bf16

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original reported failure was a transient eth-core error:
```
TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```
On reproduction, the actual failures were three transformers 5.x loader bugs, then a hardware capacity OOM after those were fixed:
```
TT_FATAL @ bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 8388608 B DRAM buffer across 8 banks,
where each bank needs to store 1048576 B, but bank size is 4273390016 B
(allocated: 4273333248 B, free: 56768 B, largest free block: 45248 B)
```

## Root cause

**Loader bugs (three):**

1. **ROPE_INIT_FUNCTIONS missing 'default' key**: transformers 5.x removed `'default'` from `ROPE_INIT_FUNCTIONS`. The model's `Param2MoERotaryEmbedding.__init__` sets `self.rope_type = "default"` when `rope_scaling` has `rope_type: "default"`, then does `ROPE_INIT_FUNCTIONS[self.rope_type]` → `KeyError: 'default'`.

2. **`_tied_weights_keys` list vs dict**: transformers 5.x changed `_tied_weights_keys` from a list of target names to a `{target: source}` dict. The model has `_tied_weights_keys = ["lm_head.weight"]` (list). `get_expanded_tied_weights_keys` calls `.keys()` on it → `AttributeError: 'list' object has no attribute 'keys'`.

3. **`moe_infer` D2H/numpy under XLA tracing**: `moe_infer` calls `.cpu().numpy()` on `tokens_per_expert`, then uses numpy scalars (`start_idx`, `end_idx`) in control flow. Under Dynamo/XLA tracing this becomes `FakeTensor + ndarray` → `AttributeError: 'ndarray' object has no attribute 'add'`.

**Hardware capacity:**
After fixing the loader bugs, the model compiles and loads its 17B BF16 weights (~34 GB) into device DRAM. The p150b has ~32 GB (8 banks × ~4 GB), which is completely consumed by model weights. When inference tries to copy the input tensor to device, the 8 MB allocation fails with OOM (only 56 KB free per bank).

## Fix

**In tt_forge_models** (`param2_17b_a2_4b_thinking/causal_lm/pytorch/loader.py`), three loader fixes on branch `remediation/param2_17b_a2_4b_thinking-causal_lm-pytorch-single_device-inference`:

1. Inject `is_torch_fx_available` shim (removed from transformers 5.x) for standalone compatibility.

2. Add `'default'` key to `ROPE_INIT_FUNCTIONS` computing vanilla base^(-2i/d) RoPE before model loads.

3. Use `get_class_from_dynamic_module` to patch `Param2MoEForCausalLM._tied_weights_keys` from list `["lm_head.weight"]` to dict `{"lm_head.weight": "model.word_embeddings.weight"}`.

4. Replace `Param2MoESparseMoeBlock.moe_infer` with a static per-expert masked matmul (no D2H/numpy) on the class via `importlib.import_module`.

**In tt-xla** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`) on branch `remediation/param2_17b_a2_4b_thinking-causal_lm-pytorch-single_device-inference`:

Added `KNOWN_FAILURE_XFAIL` for `param2_17b_a2_4b_thinking/causal_lm/pytorch-param2_17b_a2_4b_thinking-single_device-inference` with OOM reason.

## Verification
- pytest exit: FAIL (OOM after compilation — 1714.24s)
- Hardware:    blackhole-p150b
- Duration:    1714.24s (0:28:34)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/param2_17b_a2_4b_thinking/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | df7d8917e |
| tt-forge-models | f5d92b9d64 |
