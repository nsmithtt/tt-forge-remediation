# Remediation Summary: lumimaid_gguf-causal_lm-pytorch-V0_2_70B_HERETIC_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lumimaid_gguf/causal_lm/pytorch-V0_2_70B_HERETIC_I1_GGUF-single_device-inference]

## Result
XFAIL — 70B model dequantized to BF16 requires ~140 GB DRAM, exceeding single p150b capacity of 96 GB

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-missing-variant-in-gguf-files-map

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Converting and de-quantizing GGUF tensors...:  89%|████████▉ | 647/724 [03:45<00:31,  2.42it/s]

## Root cause
Two bugs in the loader, plus a hardware capacity ceiling:

1. **Loader bug — missing `_GGUF_FILES` entry**: `ModelVariant.LUMIMAID_V0_2_70B_HERETIC_I1_GGUF` was added to `_VARIANTS` and `ModelVariant` by commit `3df4601a0f` but then inadvertently dropped from `_GGUF_FILES` by the subsequent commit `6b19c66987` (which added the 8B variant and reconstructed the dict). This caused a `KeyError` on `self._GGUF_FILES[self._variant]` before the model even began loading.

2. **Loader bug — unsafe `apply_chat_template`**: `load_inputs` called `tokenizer.apply_chat_template` unconditionally; GGUF tokenizers may have `chat_template=None`, which raises a `jinja2` error at runtime.

3. **Hardware capacity ceiling**: After the loader fixes are applied, the model (`mradermacher/Lumimaid-v0.2-70B-heretic-i1-GGUF`, Q4_K_M imatrix quantization) dequantizes to BF16 consuming ~140 GB CPU RAM and would require ~140 GB device DRAM. A single p150b provides 96 GB DRAM, so the model cannot fit. The failure at 89% of the CPU-side dequantization progress bar (647/724 tensors, 3:45 elapsed) is consistent with either a machine RAM exhaustion or CI process kill during the ~5-minute dequantization of the ~40 GB GGUF file. This is a hardware class limit, not a compiler bug.

## Fix
**Loader fixes in `tt_forge_models`** (`remediation/lumimaid_gguf-causal_lm-pytorch-V0_2_70B_HERETIC_I1_GGUF-single_device-inference` branch, commit `69de50212e`):

- `lumimaid_gguf/causal_lm/pytorch/loader.py`: Added `ModelVariant.LUMIMAID_V0_2_70B_HERETIC_I1_GGUF: "Lumimaid-v0.2-70B-heretic.i1-Q4_K_M.gguf"` to `_GGUF_FILES`; guarded `apply_chat_template` with `if self.tokenizer.chat_template is not None`.
- `lumimaid_gguf/causal_lm/pytorch/requirements.txt`: Created with `gguf>=0.10.0`.

**Test config in `tt-xla`** (`remediation/lumimaid_gguf-causal_lm-pytorch-V0_2_70B_HERETIC_I1_GGUF-single_device-inference` branch, commits `0c7faebda` + `26aa023ac`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this test with explanation of the hardware capacity ceiling.

## Verification
- pytest exit: not-run (hardware capacity ceiling; would OOM during model loading)
- Hardware:    blackhole-p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/lumimaid_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/lumimaid_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 26aa023ac6ad265479f95e604882d97f9e681f1d |
| tt-forge-models | 69de50212eb2abeac6ff2029edc599353fb087c2 |
