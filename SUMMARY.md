# Remediation Summary: next_ocr_gguf-image_text_to_text-pytorch-i1_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[next_ocr_gguf/image_text_to_text/pytorch-i1_Q4_K_M-single_device-inference]

## Result
FAIL — Tier B pjrt-device-to-host-transfer: .cpu() on TT tensor inside Dynamo-compiled Qwen3VL position embedding method triggers INTERNAL Error code: 13

## Stack layer
loader, tt-xla

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
Original: E   AttributeError: 'Qwen3VLConfig' object has no attribute 'num_hidden_layers'
Reproduced as: ValueError: GGUF model with architecture qwen3vl is not supported yet.

After loader fix:
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Stack context:
  modeling_qwen3_vl.py:778: pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
  loader.py:46: in _patched_fast_pos
  torch/_dynamo/eval_frame.py:1044: in _fn
    return fn(*args, **kwargs)
  torch_xla/_dynamo/dynamo_bridge.py:826: torch_xla.sync()

## Root cause
Two bugs existed in the original loader:

1. **Loader bug — GGUF arch not registered**: transformers does not register
   `qwen3vl` in `GGUF_SUPPORTED_ARCHITECTURES`, so `AutoModelForImageTextToText
   .from_pretrained(gguf_repo, gguf_file=...)` raises `ValueError`. The original
   loader had a module-level GGUF patch, but it was being overridden at import
   time by alphabetically-later loaders whose narrow-signature wrappers called the
   real `load_gguf_checkpoint` directly, bypassing the registration. This is the
   root cause of both the original `AttributeError` (in CI, another loader had
   partially applied the arch registration) and the `ValueError` (in clean
   reproduction).

2. **Loader bug — GGUF missing vision encoder**: The `mradermacher/next-ocr-i1-GGUF`
   file ships only LM weights (396 `blk.*` tensors + `output_norm.weight` +
   `token_embd.weight`); the Qwen3VL vision encoder tensors are entirely absent.
   Loading from the GGUF would give a randomly-initialized vision encoder, making
   image_text_to_text testing invalid.

   Fix: load directly from `thelamapi/next-ocr` (the full fine-tuned VL model,
   hidden_size=4096, 36 text layers, 27 vision layers — Qwen3-VL-8B based).

3. **Tier B compiler bug — pjrt-device-to-host-transfer**: After the loader fix,
   Qwen3VL's `fast_pos_embed_interpolate`, `rot_pos_emb`, `get_rope_index`, and
   `get_image_features` methods call `.tolist()` on `grid_thw`/`input_ids` tensors
   for Python control flow. The patch applies `.cpu()` on these metadata tensors
   before the calls. However, `fast_pos_embed_interpolate` is reached inside a
   Dynamo-compiled graph (`eval_frame._fn`), and `grid_thw.cpu()` inside a Dynamo
   trace inserts a device-to-host transfer node in the XLA graph. The TT PJRT
   runtime does not support this transfer, failing with INTERNAL Error code: 13.
   This is the same Tier B bug seen in other Qwen3-VL models (bug fingerprint:
   `pjrt-device-to-host-transfer`).

## Fix
**Applied (loader layer)**: `tt_forge_models/next_ocr_gguf/image_text_to_text/pytorch/loader.py`
- Replaced the broken GGUF-loading approach with direct loading from `thelamapi/next-ocr` (the base VL model).
- Added `_patch_qwen3vl_for_tt_device()` — patches `fast_pos_embed_interpolate`, `rot_pos_emb`, `get_rope_index`, `get_image_features` to move metadata tensors to CPU before `.tolist()` calls.
- Added standard Qwen VL pixel limits (`min_pixels=56*56`, `max_pixels=13*28*1280`).
- Set default `torch_dtype=torch.bfloat16`.

**Not fixed (Tier B)**: The residual `pjrt-device-to-host-transfer` failure requires the TT PJRT runtime to support device-to-host tensor transfers for intermediate tensors during XLA graph execution. `@torch.compiler.disable` on `fast_pos_embed_interpolate` would prevent Dynamo from tracing into the patched function and avoid the transfer, but that workaround is explicitly forbidden per remediation rules.

## Tier B justification
`pjrt-device-to-host-transfer`: new-infrastructure. The TT PJRT layer does not implement device-to-host transfer for intermediate tensors inside compiled XLA graphs. Any `.cpu()` call on a TT tensor within Dynamo's trace path results in a graph node that triggers `torch_xla.sync()` failing with INTERNAL Error code: 13. Fixing this requires implementing the transfer path in the TT PJRT runtime.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    117.28s (1:57)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/next_ocr_gguf/image_text_to_text/pytorch/loader.py` (tt-forge-models remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b9664fa1fcfb8cc4c0634f6cbbaf6c496486306f |
| tt-forge-models | cd6d662a2e01fee5340b7e5bdde79ec115bf5bc0 |
