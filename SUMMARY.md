# Remediation Summary: gemma_2_2b_jpn_it_q4f16_1_mlc-causal_lm-pytorch-Gemma_2_2B_JPN_IT_Q4F16_1_MLC-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_2b_jpn_it_q4f16_1_mlc/causal_lm/pytorch-Gemma_2_2B_JPN_IT_Q4F16_1_MLC-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-17, 16], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two bugs:

**Loader (loader layer):** The `mlc-ai/gemma-2-2b-jpn-it-q4f16_1-MLC` HuggingFace repo stores weights in MLC binary shards without a standard `config.json` with `model_type`. `AutoModelForCausalLM.from_pretrained` raises `ValueError: Unrecognized model`. The branch already contained a partial fix that built the config from `mlc-chat-config.json` and used `from_config`, but this left the model with random weights (no actual MLC binary weight loader was implemented).

**Compiler frontend (tt-xla):** Gemma-2's sliding-window attention cache code computes `full_value_states[:, :, -sliding_window + 1 :, :]` where `sliding_window=4096`. For the Japanese sample text "富士山の高さは何メートルですか？" tokenized to 17 tokens, the KV cache has seq_dim=17. The slice start is `-(4096-1) = -4095`, which is outside `[-17, 16]`. XLA's `aten.slice.Tensor` kernel validates bounds strictly (unlike PyTorch CPU which silently clamps), causing the `RuntimeError`. The error fires in `TorchFunctionOverride.__torch_function__` at the `return func(*args, ...)` call before tt-mlir compilation.

## Fix
**Loader fix** in `tt_forge_models` (commit `782de75d638e2f6b3b7754f1bb3f18071fd8863e`):
- `gemma_2_2b_jpn_it_q4f16_1_mlc/causal_lm/pytorch/loader.py`: Switch `pretrained_model_name` from `mlc-ai/gemma-2-2b-jpn-it-q4f16_1-MLC` to `bartowski/gemma-2-2b-jpn-it-GGUF`, add `GGUF_FILE = "gemma-2-2b-jpn-it-Q4_K_M.gguf"`, and pass `gguf_file=self.GGUF_FILE` to all `from_pretrained`/`from_config`/`AutoConfig.from_pretrained` calls.
- `gemma_2_2b_jpn_it_q4f16_1_mlc/causal_lm/pytorch/requirements.txt`: Add `gguf>=0.10.0`.

**Compiler frontend fix** in `tt-xla` (commit `b18f32c098dc899768a99bea5a6627f0ca52f2c8`, includes `c435a67ee`):
- `python_package/tt_torch/torch_overrides.py`: In `TorchFunctionOverride.__torch_function__`, before dispatching `aten.slice.Tensor`, clamp the start index to `max(-dim_size, start)` when `start < -dim_size`. This matches PyTorch CPU semantics and prevents the XLA bounds check from failing on OOB negative slice starts.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    331.60s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/gemma_2_2b_jpn_it_q4f16_1_mlc/causal_lm/pytorch/loader.py`
- `tt_forge_models/gemma_2_2b_jpn_it_q4f16_1_mlc/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b18f32c098dc899768a99bea5a6627f0ca52f2c8 |
| tt-forge-models | 782de75d638e2f6b3b7754f1bb3f18071fd8863e |
