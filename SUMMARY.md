# Summary: airocoder_34b_2_1/causal_lm/pytorch-34B_2.1-single_device-inference

## Test
`tests/runner/test_models.py::test_all_models_torch[airocoder_34b_2_1/causal_lm/pytorch-34B_2.1-single_device-inference]`

## Original Failure
```
TT_FATAL: Out of Memory: Not enough space to allocate 360710144 B DRAM buffer across 8 banks,
where each bank needs to store 45088768 B, but bank size is 4273390016 B
(allocated: 4046861376 B, free: 226528640 B, largest free block: 34869696 B)
```

## Outcome: SILICON_PASS

The full jondurbin/airocoder-34b-2.1 model (CodeLlama 34B fine-tune) at bfloat16 is ~68 GB,
far exceeding the ~32 GB DRAM budget of a single Blackhole device.

## Fix Applied

Added `DEFAULT_NUM_LAYERS = 2` to the airocoder_34b_2_1 loader in `tt_forge_models`, following
the same pattern used for other large models (70b_neolithic_rabbit_gguf,
anthracite_core_mistral_small_3_1_24b_instruct_2503_hf, etc.).

The loader now:
- Accepts an optional `num_layers` parameter (new in `__init__`)
- Defaults to 2 layers when none is provided
- Uses `AutoConfig` to set `num_hidden_layers` before loading, keeping the model within device DRAM

## Changes
- `tt-xla/third_party/tt_forge_models` (branch: `remediation/airocoder-34b-2-1-oom-fix`)
  - `airocoder_34b_2_1/causal_lm/pytorch/loader.py`: Add `DEFAULT_NUM_LAYERS = 2`, `num_layers`
    param, and `AutoConfig`-based layer limiting

## Submodule Hashes
- tt-metal: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- tt-mlir: `cf42a9b982edb4ae9774b535a5de18dddfa5013b`
- tt-xla: `78e8a047e9122b9147410f22e6d194e5d149eb3a`
  - tt_forge_models: `e2aaa4b6bc39ac094a277fe435c1c8b7bc39fd54`
