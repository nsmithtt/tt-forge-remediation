# Remediation Summary: amplify-masked_lm-pytorch-chandar-lab-AMPLIFY_350M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[amplify/masked_lm/pytorch-chandar-lab/AMPLIFY_350M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
amplify-xformers-dependency-non-contiguous-att-block

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
Actual underlying error:
```
ImportError: This modeling file requires the following packages that were not found in your environment: xformers. Run `pip install xformers`
```

## Root cause
Three cascading loader-layer bugs, all in `tt_forge_models/amplify/masked_lm/pytorch/loader.py`:

**Bug 1 — xformers import (loader)**
The AMPLIFY model from `chandar-lab/AMPLIFY_350M` uses `trust_remote_code=True`, which
fetches a custom `amplify.py` from HuggingFace. That file imports
`from xformers.ops import SwiGLU, memory_efficient_attention` at module level.
Transformers `check_imports()` validates all imports before loading the module, so the
test fails immediately because `xformers` (a CUDA-optimised library) is not installed.
The `memory_efficient_attention` function is only called when `x.is_cuda` — never on TT
hardware — but `SwiGLU` is used unconditionally for the FFN.

**Bug 2 — freqs_cis meta tensor (loader, transformers 5.x)**
Transformers 5.x `from_pretrained` uses lazy (meta-device) initialisation by default.
The model's `freqs_cis` attribute is set in `__init__` via a plain tensor assignment —
not `register_buffer` — so it is absent from `state_dict()` and never materialised from
the checkpoint. After loading, `model.freqs_cis.is_meta` is True; the first forward call
raises `NotImplementedError: Cannot copy out of meta tensor; no data!`.

**Bug 3 — non-contiguous view in _att_block (loader)**
The hub's `EncoderBlock._att_block` builds the attention result as
`sdpa(...).transpose(1, 2)` on non-CUDA (including TT/XLA) devices. `transpose`
returns a non-contiguous view. The next line calls `attn.view(B, S, H*D)`, which
requires a contiguous tensor. On CUDA, `xformers.ops.memory_efficient_attention`
always returned a contiguous tensor, masking this bug. Dynamo raises
`TorchRuntimeError: Cannot view a tensor with shape (1,44,15,64) and strides
(42240,64,2816,1) as a tensor with shape (1,44,960)`.

## Fix
All fixes are in `tt_forge_models`, `amplify/masked_lm/pytorch/loader.py`.

**Fix 1 — xformers stub**: Inject a minimal `xformers` stub into `sys.modules` before
any model loading. The stub provides:
- `xformers.ops.SwiGLU` — pure-PyTorch `nn.Module` with `w12` (fused) + `w3` weights
  matching the pretrained checkpoint's parameter layout.
- `xformers.ops.memory_efficient_attention` — falls through to
  `F.scaled_dot_product_attention` (never reached on TT hardware, since `x.is_cuda`
  is False).

**Fix 2 — freqs_cis materialisation**: After `from_pretrained`, check if `model.freqs_cis`
is a meta tensor. If so, recompute it from `model.config` using the same formula as
`precompute_freqs_cis` in the hub's `rotary.py`.

**Fix 3 — _att_block contiguous patch**: After loading, replace `EncoderBlock._att_block`
on the hub's class (accessed via `type(model.transformer_encoder[0])`) with a version
that calls `.contiguous()` on `attn` before `.view()` in the non-CUDA path.

**Additionally**: switched `AutoModelForMaskedLM` → `AutoModel` (the checkpoint's
`config.json` maps only `AutoModel` in `auto_map`), and converted the tokenizer's binary
attention mask to the additive format expected by the AMPLIFY forward method.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    112.83s
- Tier A attempts: N/A

## Files changed
- `amplify/masked_lm/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7d6ef3413ea25293184a10a7951c71320f266bf4 |
| tt-forge-models | e85b1c31d8383a4b3caa4059a4f92a9b98490605 |
