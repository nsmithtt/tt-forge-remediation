# Anthracite-Core Mistral-Small-3.1-24B-Instruct-2503-HF Remediation Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[anthracite_core_mistral_small_3_1_24b_instruct_2503_hf/causal_lm/pytorch-24B_Instruct_2503_HF-single_device-inference]`

## Failure
```
TT_FATAL: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4196976832 B, free: 76413184 B, largest free block: 37030336 B)
```

## Root Cause

The full `anthracite-core/Mistral-Small-3.1-24B-Instruct-2503-HF` model (40 transformer layers) exhausts the single Blackhole device's DRAM (~32 GB total across 8 banks). With 31.3 GB allocated for model weights and compiled buffers, only ~583 MB remains across all banks with the largest contiguous free block (~35 MB per bank) smaller than the 40 MB needed for the final allocation.

The loader accepted a `num_layers` parameter but defaulted to `None`, which caused the full 40-layer model to load every time. There was no `DEFAULT_NUM_LAYERS` to limit the model to a testable size.

## Fix

**Repository:** `tt-forge-models`
**Branch:** `remediation/anthracite-mistral-small-24b-oom-fix`
**Commit:** `1ba886e0834c34601b5a0a05c7213fb6255d37fb`

Added `DEFAULT_NUM_LAYERS = 2` to the `ModelLoader` class and updated `__init__` to use this default when `num_layers` is not specified. With only 2 layers, the model's compiled footprint is a small fraction of device DRAM, leaving ample space for activations and intermediate buffers.

```python
DEFAULT_NUM_LAYERS = 2

def __init__(self, variant=None, num_layers=None):
    ...
    self.num_layers = num_layers if num_layers is not None else self.DEFAULT_NUM_LAYERS
```

## Result
Test passes: `1 passed in 142.52s (0:02:22)`

## Submodule Hashes
- tt-xla: `ae67c1ba74a2b519dede13007eb4fe4610defeba`
- tt-mlir: `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- tt-metal: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- tt-forge-models: `1ba886e0834c34601b5a0a05c7213fb6255d37fb`
