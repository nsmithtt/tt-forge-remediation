# Remediation Summary: dolphin-2-5-mixtral-gguf-causal-lm-pytorch-dolphin-2-5-mixtral-8x7b-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dolphin_2_5_mixtral_gguf/causal_lm/pytorch-DOLPHIN_2_5_MIXTRAL_8X7B_GGUF-single_device-inference]

## Result
XFAIL — Mixtral 8x7B (~46.7 B params, ~93 GB BF16) exceeds single-device DRAM on all TT hardware (n150: 12 GB, p150b: 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-46b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Local reproduction shows:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(Another GGUF loader in the test session monkey-patches `load_gguf_checkpoint` without `**kwargs`; transformers 5.x passes `model_to_load=dummy_model` as a keyword argument.)

The originally reported failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```
(gguf package was not installed at the time of the CI failure; it has since been added to tt-xla dev requirements.)

## Root cause
Three compounding issues, all in the loader layer:

1. **Missing `requirements.txt`** (`gguf>=0.10.0`): the loader had no `requirements.txt`, so `gguf` was not a declared dependency. Fixed by adding `requirements.txt`.

2. **`_patched_load_gguf_checkpoint` missing `**kwargs` in 26 GGUF loaders** (pre-existing fix on remediation branch, commit `e59922f550`): transformers 5.x `load_gguf_checkpoint()` added `model_to_load=None` kwarg. Loader patches with fixed signatures cause `TypeError` for any subsequently-collected GGUF model test.

3. **Missing Mixtral GGUF architecture support** (pre-existing fixes on remediation branch, commits `e13eed917b`, `d03b1d1500`, `ef39dd620c`): Mixtral GGUFs use `general.architecture = "llama"` but store per-expert MoE tensors as `blk.N.ffn_{gate,up,down}.K.weight` (not the stacked format). Needed: tokenizer registration, `MixtralTensorProcessor` to accumulate per-expert slices, and `get_gguf_hf_weights_map` patching.

Even with all loader fixes applied, the model cannot run on any single TT device:
- Mixtral 8x7B has ~46.7B parameters
- BF16 footprint: 46.7 × 10⁹ × 2 bytes ≈ **93.4 GB**
- n150 DRAM: 12 GB; p150b DRAM: 24 GB
- Ratio: 93.4 GB / 24 GB ≈ 3.9× over capacity

The prior FAIL report (Tier B, SIGSEGV in `partition_fx_graph_for_cpu_fallback`) was caused by the compilation infrastructure crashing before reaching silicon, likely as a pre-OOM symptom of the model being too large to partition. Either way, the test cannot pass on any current TT single-device hardware.

## Fix
Loader fixes (all on remediation branch `e563636d03` in tt-forge-models):
- Added `dolphin_2_5_mixtral_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`
- 26 GGUF loaders: `_patched_load_gguf_checkpoint` signature + call site updated to accept and forward `**kwargs`
- `loader.py`: `_patch_mixtral_gguf_support()` implements `MixtralTensorProcessor` (per-expert tensor accumulation), tokenizer registration, and `get_gguf_hf_weights_map` patching

Test config update (tt-xla, commit `a06f80d76`):
- Added `dolphin_2_5_mixtral_gguf/causal_lm/pytorch-DOLPHIN_2_5_MIXTRAL_8X7B_GGUF-single_device-inference: KNOWN_FAILURE_XFAIL` to `test_config_inference_single_device.yaml`

## Verification
- pytest exit: not-run (KNOWN_FAILURE_XFAIL; model too large to load onto silicon)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/dolphin_2_5_mixtral_gguf/causal_lm/pytorch/requirements.txt` (added)
- `tt_forge_models/dolphin_2_5_mixtral_gguf/causal_lm/pytorch/loader.py` (Mixtral GGUF patching)
- 26 other GGUF loader files (`**kwargs` fix)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a06f80d76114c74010d4bc685cea2ac392cf3e7e |
| tt-forge-models | e563636d03f5d4a67b76870e9736b17be5852434 |
