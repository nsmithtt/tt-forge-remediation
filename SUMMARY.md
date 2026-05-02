# Remediation Summary: mradermacher_pg67a_w_serum_1b_i1_gguf-causal_lm-pytorch-PG67A_W_Serum_1B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_pg67a_w_serum_1b_i1_gguf/causal_lm/pytorch-PG67A_W_Serum_1B_i1_GGUF-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -511)

## Root cause
Three bugs in sequence:

**Bug 1 — GGUF loader global patching TypeError (loader bug)**
26 loaders in `tt_forge_models` patched `load_gguf_checkpoint` at module import time with a broken signature `(gguf_path, return_tensors=False)` missing the `model_to_load` kwarg added in transformers 5.x. These module-level patches polluted global state, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` for any GGUF model loaded in a full pytest session after those loaders were imported.

**Bug 2 — GGUF tokenizer no chat_template (loader bug)**
The PG67A-W-Serum-1B GGUF tokenizer has `chat_template=None`. The loader unconditionally called `apply_chat_template`, raising `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`.

**Bug 3 — aten.slice OOB start for sliding-window attention (tt-xla bug)**
PG67A-W-Serum-1B uses `gemma3_text` architecture with `sliding_window=512`. Its attention bias computation produces `attn_bias[:, :, -511:, :]` where the target dim has size 1. PyTorch eager silently clamps OOB slice indices; the XLA lazy backend raises `RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -511)`. The fix clamps start/end in `TorchFunctionOverride.__torch_function__` before the op reaches XLA.

## Fix
**Loader fixes** (in `tt_forge_models` via tt-xla's `third_party/tt_forge_models` submodule, branch `remediation/mradermacher_pg67a_w_serum_1b_i1_gguf-aten-slice-oob`, commit `fcda01af1b4046d51b97b6b5a138fd6ea3184acc`):
- 26 GGUF loaders: updated `_patched_load_gguf_checkpoint` signature to `(*args, **kwargs)` forwarding to the original function
- `mradermacher_pg67a_w_serum_1b_i1_gguf/causal_lm/pytorch/loader.py`: guard `apply_chat_template` with `if self.tokenizer.chat_template is not None:`

**Compiler frontend fix** (in `tt-xla`, branch `remediation/mradermacher_pg67a_w_serum_1b_i1_gguf-aten-slice-oob`, commits `6fb2b952f` and `46ef59938`):
- `python_package/tt_torch/torch_overrides.py`: In `TorchFunctionOverride.__torch_function__`, intercept `aten.slice.Tensor` and clamp any integer start/end that is `< -size` to `-size`, matching PyTorch eager semantics
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Mark `mradermacher_pg67a_w_serum_1b_i1_gguf/causal_lm/pytorch-PG67A_W_Serum_1B_i1_GGUF-single_device-inference` as `EXPECTED_PASSING`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    ~45 min (full configure+build+run cycle)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer update)
- 26 GGUF loader files in `tt_forge_models` (signature fix)
- `tt_forge_models/mradermacher_pg67a_w_serum_1b_i1_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 46ef59938753fa44d4514e3a40163d9f81fc1f6b |
| tt-forge-models | fcda01af1b4046d51b97b6b5a138fd6ea3184acc |
