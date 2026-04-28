# Remediation Summary: fun_asr_nano-speech_recognition-pytorch-2512_vllm-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fun_asr_nano/speech_recognition/pytorch-2512_vllm-single_device-inference]

## Result
SILICON_PASS — loader redesigned to use forward_export (audio encoder only) and fixed to run in float32 to match SenseVoice's hardcoded float32 masks

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
funasr-sense-voice-sequence-mask-float32-bfloat16-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `hcj_name = re.sub("(?<=HANGUL )(\w+)", ...)` — jamo Korean post-processing inside FunASR's autoregressive `AutoModel.inference()` pipeline, which returns `List[Dict]` with text strings rather than tensors.

After fixing the inference path: `RuntimeError: Input type (torch.FloatTensor) and weight type (CPUBFloat16Type) should be the same` — the TorchDynamicLoader forces `dtype_override=torch.bfloat16` on any loader with that parameter in its signature, but SenseVoice's `sequence_mask()` hardcodes `dtype=torch.float32`; multiplying bfloat16 activations by the float32 mask promoted the result to float32, which then mismatched the bfloat16 `fsmn_block` (Conv1d) weights.

## Root cause
Two bugs in the loader layer:

1. **Wrong inference method**: The original loader called `AutoModel.inference()`, which runs the full autoregressive ASR pipeline (encoder → LLM decode → jamo post-processing → text). This is not compilable and returns Python strings rather than tensors. The correct interface is `FunASRNano.forward_export(speech, speech_lengths)`, which is the pure-tensor audio-encoder path used for export/compilation.

2. **bfloat16/float32 dtype mismatch**: `TorchDynamicLoader` detects `dtype_override` in the loader's signature and passes `dtype_override=torch.bfloat16`, causing the model to be converted to bfloat16. However, `funasr/models/sense_voice/model.py:sequence_mask()` always returns `dtype=torch.float32`. In `SenseVoiceSANMAttention.forward_fsmn()`, the expression `inputs * mask` (bfloat16 × float32) promotes `inputs` to float32 via PyTorch type promotion, and the resulting float32 tensor then fails against the bfloat16 `fsmn_block` Conv1d weights.

## Fix
All changes in `tt-xla/third_party/tt_forge_models/fun_asr_nano/speech_recognition/pytorch/`:

**loader.py** (commits `9cf304cdc0`, `0f1fe9b48b`, `698073c34e` on branch `remediation/fun_asr_nano-speech_recognition-pytorch-2512_vllm-single_device-inference`):
- Replaced `FunASRNanoWrapper.forward` calling `AutoModel.inference()` with a proper wrapper around `FunASRNano.forward_export(speech, speech_lengths)` — tensor-in/tensor-out audio encoder path.
- Changed `load_inputs` to use `extract_fbank(audio_array, frontend=frontend)` from `funasr.utils.load_utils` (free function, not a method) to produce properly shaped FBANK feature tensors.
- Added pre-registration of `FunASRNano` via `spec_from_file_location` on funasr's bundled `model.py`, since ModelScope download dir ships no `model.py` and funasr's trust_remote_code path would otherwise load a wrong `model` module from sys.path.
- Added `llm_conf: {init_param_path: "Qwen/Qwen3-0.6B"}` to AutoModel kwargs to fix the missing `Qwen/` org prefix in the bundled `config.yaml`.
- Added `model.to(torch.float32)` in `load_model` to homogenize the mixed-precision state (LLM loads as bfloat16 from `llm_dtype: bf16` in config).
- Removed `dtype_override` from `load_model` and `load_inputs` signatures so `TorchDynamicLoader` does not force bfloat16 conversion; the model stays float32 throughout, which is consistent with SenseVoice's float32 masks.

**requirements.txt**: Added `tiktoken` (required by SenseVoiceTokenizer via whisper_lib) and pinned `torchaudio` to the CPU build.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    224.87s (0:03:44)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/fun_asr_nano/speech_recognition/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/fun_asr_nano/speech_recognition/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0b2012865778081d48a09ea22f20bce0b1eda469 |
| tt-forge-models | 698073c34e64e962e70fd6fb39b5f04fe49e4141 |
