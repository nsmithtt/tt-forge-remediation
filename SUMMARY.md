# Remediation Summary: kani_tts_gguf-pytorch-400M_EN_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kani_tts_gguf/pytorch-400M_EN_Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-missing-requirements

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

## Root cause
Three cascading loader bugs, each hidden behind the previous:

1. **Missing gguf requirement** (`gguf-missing-requirements`): `kani_tts_gguf/pytorch/requirements.txt` did not exist, so `gguf>=0.10.0` was not installed. transformers raises `ImportError` when attempting to load a GGUF checkpoint without the `gguf` package.

2. **lfm2 tokenizer converter missing** (`gguf-lfm2-tokenizer-converter-missing`): Once gguf was installed, the tokenizer load failed with `KeyError: 'lfm2'`. Kani-TTS uses the LFM2 architecture, but `transformers.integrations.ggml.GGUF_TO_FAST_CONVERTERS` has no entry for `'lfm2'`. LFM2 uses a GPT2-style BPE tokenizer (`tokenizer.ggml.model = "gpt2"`), so `GGUFGPTConverter` is the correct mapping.

3. **Cross-loader model_to_load TypeError** (`gguf-load-checkpoint-model-to-load-kwarg`): After fixing the tokenizer, model loading failed with `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Another GGUF loader imported during pytest collection installed a narrow-signature wrapper on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. transformers 5.2.0 calls it with `model_to_load=dummy_model`, which the stale wrapper rejects. The fix uses a DFS closure/globals walker to recover the real function just-in-time before `from_pretrained`.

4. **Lfm2HybridConvCache pytree error** (pre-emptive): LFM2 returns `Lfm2HybridConvCache` when `use_cache=True`, which is not a registered pytree node and causes comparison failures. Added `use_cache=False` to `load_inputs`.

## Fix
All changes in `kani_tts_gguf/pytorch/` in tt-forge-models on branch
`remediation/kani_tts_gguf-pytorch-400M_EN_Q4_K_M-single_device-inference`:

- **`kani_tts_gguf/pytorch/requirements.txt`** (new file): `gguf>=0.10.0`
- **`kani_tts_gguf/pytorch/loader.py`**:
  - Import `GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter` and register `"lfm2"` at module level.
  - Add `_get_real_load_gguf_checkpoint()` DFS walker to bypass the broken cross-loader patch.
  - In `load_model`, temporarily restore the real `load_gguf_checkpoint` around `from_pretrained`.
  - Add `inputs["use_cache"] = False` in `load_inputs` to avoid `Lfm2HybridConvCache` output.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    142.81s (0:02:22)
- Tier A attempts: N/A

## Files changed
- `kani_tts_gguf/pytorch/requirements.txt` (created)
- `kani_tts_gguf/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | b82ee25aae02cc52d6d0355395f85cb5c4683a3e |
