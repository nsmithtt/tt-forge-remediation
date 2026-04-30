# Remediation Summary: glm_4_voice-causal_lm-pytorch-glm-4-voice-9b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_voice/causal_lm/pytorch-glm-4-voice-9b-single_device-inference]

## Result
SILICON_PASS — three loader bugs fixed; test passes in 186.80s on n150

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
chatglm-tokenizer-position-ids-unpadded, chatglm-inplace-bool-unsqueeze-xla, chatglm-all-tied-weights-keys-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The CI failure message (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__
attribute`) was the last line of pytest output, not the root cause. Three real loader bugs were
found in sequence:

1. `AttributeError: 'ChatGLMForConditionalGeneration' object has no attribute 'all_tied_weights_keys'`
2. `RuntimeError: shape '[-1, 1, 137, 32, 2]' is invalid for input of size 576` (in TorchScript
   `apply_rotary_pos_emb`)
3. `RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLABoolType` (during
   AOT autograd compilation of `get_masks`)

## Root cause
All three bugs are in the loader (remote code compatibility with transformers 5.x and tt-xla):

**Bug 1** — `all_tied_weights_keys` missing: `ChatGLMForConditionalGeneration.__init__` does not
call `self.post_init()`. In transformers 5.x, `post_init()` initialises `all_tied_weights_keys`
via `get_expanded_tied_weights_keys()`. Without it, `_finalize_model_loading()` raises
AttributeError.

**Bug 2** — tokenizer `position_ids` mismatch: `AutoTokenizer` returns `position_ids` of shape
`(1, original_seq_len)` covering only the 9 unpadded tokens. After padding `input_ids` and
`attention_mask` to 137 tokens, the model's `forward()` branches to
`rotary_pos_emb = rotary_pos_emb[position_ids]` (indexed by only 9 positions) rather than
`rotary_pos_emb[None, :seq_length]` (137 positions). The resulting `(1, 9, 32, 2)` rope cache
fails the `.view(-1, 1, 137, 32, 2)` in `apply_rotary_pos_emb`.

**Bug 3** — in-place unsqueeze on bool tensor: `ChatGLMModel.get_masks()` ends with
`full_attention_mask.unsqueeze_(1)` on an `XLABoolType` tensor. During AOT autograd graph
capture (`PropagateUnbackedSymInts`), `aten.unsqueeze_` is dispatched through
`functional_tensor.__torch_dispatch__` to the XLA implementation, which rejects in-place writes
on bool tensors (`Check failed: xtensor: Input tensor is not an XLA tensor: XLABoolType`).

## Fix
All three fixes are in `glm_4_voice/causal_lm/pytorch/loader.py` in the
`worktree-aknezevic+hf-bringup_1023-1` branch of `tenstorrent/tt-forge-models`:

1. **post_init patch**: Before `from_pretrained`, load `ChatGLMForConditionalGeneration` via
   `get_class_from_dynamic_module` and patch `__init__` to call `self.post_init()` if
   `all_tied_weights_keys` is not set.

2. **Drop position_ids**: In `load_inputs()`, call `inputs.pop("position_ids", None)` after
   padding. With `position_ids=None`, the model uses `rotary_pos_emb[None, :seq_length]` which
   covers all 137 padded positions.

3. **get_masks replacement**: Load `ChatGLMModel` via `get_class_from_dynamic_module` and
   replace `get_masks` with a functionally identical version using out-of-place `.tril()` and
   `.unsqueeze(1)` instead of `.tril_()` and `.unsqueeze_(1)`.

Commits on `remediation/glm_4_voice-causal_lm-pytorch-glm-4-voice-9b-single_device-inference`:
- `c07830c925` — glm_4_voice: drop position_ids from padded prefill inputs (includes all_tied_weights_keys + use_cache patches)
- `780c3347e3` — glm_4_voice: patch get_masks to use out-of-place unsqueeze on bool tensor

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    186.80s (0:03:06)
- Tier A attempts: N/A

## Files changed
- `glm_4_voice/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | d4f3ab29f399ec55335cb808a07615a128d71f91 |
| tt-forge-models | 780c3347e348a3de5af77e40dff2c40ae232ca2d |
