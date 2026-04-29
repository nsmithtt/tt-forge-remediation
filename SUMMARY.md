# Remediation Summary: beepo_22b_gguf/causal_lm/pytorch-22B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[beepo_22b_gguf/causal_lm/pytorch-22B_GGUF-single_device-inference]

## Result
XFAIL — 22B BF16 model (~44 GB) exceeds single-device DRAM capacity (p150b ~32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-22b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

In local reproduction with the hf-bringup-22 branch, the failure surfaced as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
because other GGUF model loaders imported during pytest collection had patched
`load_gguf_checkpoint` with a broken fixed signature (missing `**kwargs`).

## Root cause
Two loader-layer bugs, and one hardware capacity ceiling:

1. **Missing requirements.txt** (`loader` layer): `beepo_22b_gguf/causal_lm/pytorch/` had no
   `requirements.txt`, so `gguf` was not installed by `RequirementsManager` in a clean CI
   environment, causing `is_gguf_available()` to return `False` → `ImportError`.

2. **Stale `PACKAGE_DISTRIBUTION_MAPPING` cache** (`loader` layer): When `gguf` is installed at
   runtime by `RequirementsManager` (because it was absent from the base venv), transformers'
   `PACKAGE_DISTRIBUTION_MAPPING` cache does not update automatically, so `is_gguf_available()`
   still returns `False` even after installation. The fix invalidates this cache.

3. **Broken `_patched_load_gguf_checkpoint` signatures** (`loader` layer): 26 other GGUF loaders
   in tt_forge_models had monkey-patched `load_gguf_checkpoint` with a fixed signature
   `(gguf_path, return_tensors=False)` missing `**kwargs`. Transformers 5.2.0 added
   `model_to_load=None` to this call site; any co-collected GGUF test that reached
   `load_gguf_checkpoint` after collection would receive a `TypeError`.

4. **Hardware capacity ceiling**: After all loader fixes, the model loads successfully but
   OOMs on silicon. `concedo/Beepo-22B-GGUF` is Q4_K_M quantized but is dequantized to BF16
   at load time by transformers: 22B × 2 bytes ≈ 44 GB >> p150b DRAM (~32 GB usable).
   Confirmed OOM: `Out of Memory: Not enough space to allocate 201326592 B DRAM buffer across
   8 banks, where each bank needs to store 25165824 B, but bank size is 4273390016 B
   (allocated: 4081641664 B, free: 191748352 B)`.

## Fix
**In `tt_forge_models` (branch `remediation/beepo_22b_gguf-causal_lm-pytorch-22B_GGUF-single_device-inference`):**

- `beepo_22b_gguf/causal_lm/pytorch/requirements.txt` (new file): `gguf>=0.10.0`
- `beepo_22b_gguf/causal_lm/pytorch/loader.py`: Added `_fix_gguf_version_detection()` static
  method that patches `PACKAGE_DISTRIBUTION_MAPPING` and clears the `is_gguf_available` LRU
  cache after runtime gguf installation. Called from both `_load_tokenizer` and `load_model`.
- All 26 other GGUF loaders: Updated `_patched_load_gguf_checkpoint` from
  `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, **kwargs)` and
  forwarded `**kwargs` to `_orig_load_gguf_checkpoint`.

**In `tt-xla` (branch `remediation/beepo_22b_gguf-causal_lm-pytorch-22B_GGUF-single_device-inference`):**

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added
  `KNOWN_FAILURE_XFAIL` entry for this test (hardware capacity ceiling).
- `third_party/tt_forge_models`: Bumped submodule pointer to the fixed tt_forge_models commit.

## Verification
- pytest exit: FAIL (OOM on silicon — expected, XFAIL config added)
- Hardware:    blackhole-p150b
- Duration:    658.43s (0:10:58)
- Tier A attempts: N/A

## Files changed
- `beepo_22b_gguf/causal_lm/pytorch/requirements.txt` (new)
- `beepo_22b_gguf/causal_lm/pytorch/loader.py`
- 26 other GGUF loader files (broken `_patched_load_gguf_checkpoint` signature)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f4cbda3fa430795200f7c62e132e8c4b8cb26de5 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
