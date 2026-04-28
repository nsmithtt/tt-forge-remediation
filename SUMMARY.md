# Remediation Summary: glm_ocr_gguf-image_to_text-pytorch-glm_ocr_q8_0-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glm_ocr_gguf/image_to_text/pytorch-glm_ocr_q8_0-single_device-inference]

## Result
FAIL — Tier B pjrt-device-to-host-transfer: TT PJRT backend cannot convert device scalar tensor to Python int in vision encoder `rot_pos_emb`

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
  File "transformers/models/glm_ocr/modeling_glm_ocr.py", line 586, in rot_pos_emb
      hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
```

The model loads and begins forward pass, then fails when the vision encoder's
`rot_pos_emb` function calls `torch.arange(h)` where `h` is a scalar tensor
residing on the TT device. The PJRT backend cannot transfer it to host to
construct the range, producing `INTERNAL: Error code: 13`.

## Root cause

In `transformers/models/glm_ocr/modeling_glm_ocr.py:583–610`, `rot_pos_emb`
iterates over `grid_thw` (a `(N, 3)` tensor of `[t, h, w]` grid dimensions on
device) with:

```python
for t, h, w in grid_thw:
    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    ...
```

Each of `h` and `w` is a zero-dimensional device tensor (TT silicon). `torch.arange`
requires a Python int, which requires a device-to-host scalar transfer.
The TT PJRT backend does not implement this transfer path, raising
`INTERNAL: Error code: 13`.

This is in the tt-xla PJRT layer — the backend needs to support scalar device→host
transfers so that Python operators (like `range`, `torch.arange`) can consume
device tensors as integer arguments.

Four loader-layer bugs were fixed before reaching this blocker (see Files changed).

## Fix
Proposed fix: implement device-to-host scalar tensor transfer in the TT PJRT
backend (`tt-xla`), or alternatively in the tt-mlir tensor bridge layer, so
that `.item()` / `int()` / `__index__` on a zero-dimensional TT device tensor
correctly transfers the value to host CPU before use.

The fix would live in `tt-xla` in the PJRT transfer path, likely in
`python_package/tt_torch/` or in the PJRT plugin C++ layer that handles
`PjRtBuffer::ToLiteralSync` for scalar buffers.

## Tier B justification
**new-infrastructure**: The PJRT device-to-host scalar transfer path does not
exist in the TT backend. Implementing it requires adding a new transfer mechanism
in the PJRT plugin (C++ and/or Python bridge), touching the runtime data-movement
layer — this is new infrastructure, not a scoped single-function patch.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    not-run (stopped at Tier B after loader fixes confirmed)
- Tier A attempts: N/A

## Files changed
**tt_forge_models** (remediation branch `remediation/glm-ocr-gguf-image-to-text-pytorch-glm-ocr-q8-0-single-device-inference`):
- `glm_ocr_gguf/image_to_text/pytorch/loader.py` — use_fast=False, register glm4 GGUF arch, patch get_gguf_hf_weights_map, restore rope_parameters from base model
- 26 GGUF loader files (bartowski_coniccat_qwen3_5_27b_writer_gguf, daniloreddy_qwen3_5_0_8b_gguf, dmind_3_mini_i1_gguf, gpt_oss_swallow_120b_rl_gguf, gpt_oss_swallow_120b_sft_gguf, gpt_oss_swallow_20b_rl_gguf, gpt_oss_swallow_20b_sft_gguf, mradermacher_*×8, tvall43_*×2, unified_reward_flex_qwen35_27b_gguf, qwen_3_5_imatrix_gguf, noctrex_*×2, onion008_*×2, qwen_3_5_35b_claude_distilled_gguf): added `**kwargs` to `_patched_load_gguf_checkpoint` for `model_to_load` keyword compat

**tt-xla** (remediation branch `remediation/glm-ocr-gguf-image-to-text-pytorch-glm-ocr-q8-0-single-device-inference`):
- `third_party/tt_forge_models` — submodule pointer updated to remediation commit

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d836c713f6c9f62c3ecdb9a3eb884a5ebceb637b |
| tt-forge-models | d08297e986c6565d3f92d1c5bc2062065946d900 |
