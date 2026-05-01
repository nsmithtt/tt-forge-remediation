# Remediation Summary: mdlm-pytorch-OWT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mdlm/pytorch-OWT-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-flash-attn-remote-code-no-post-init

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ImportError: This modeling file requires the following packages that were not found in your environment: flash_attn. Run `pip install flash_attn`

(reported as: sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute)

## Root cause
Five cascading loader bugs in `tt_forge_models/mdlm/pytorch/loader.py`:

1. **flash_attn unconditional import**: `modeling_mdlm.py` imports `flash_attn`
   unconditionally at module top-level. Transformers 5.x `check_imports` scans
   remote-code files for package names and raises `ImportError` before the class
   is even instantiated.

2. **Missing `post_init()` call (transformers 5.x)**: `MDLM.__init__` does not
   call `self.post_init()`, which is required by transformers 5.x to initialize
   `all_tied_weights_keys`. `_finalize_model_loading` calls
   `_adjust_tied_keys_with_tied_pointers` which accesses this missing attribute.

3. **GPT-2 tokenizer missing pad_token**: `AutoTokenizer.from_pretrained("gpt2")`
   returns a tokenizer with no `pad_token`. The loader calls the tokenizer with
   `padding="max_length"`, causing `ValueError`.

4. **`modulate_fused` TorchScript / TorchDynamo name collision**: `modeling_mdlm.py`
   defines `modulate` twice. `modulate_fused` is `@torch.jit.script` and captured
   the first definition (no `unsqueeze`) at compile time, but TorchDynamo
   re-executes the function's Python source using the current module-global
   `modulate` (the second definition, which calls `.unsqueeze(1)` on shift/scale),
   producing a 4D tensor that causes `einops.rearrange` to fail.

5. **`torch.cuda.amp.autocast(dtype=bfloat16)` raises on non-bfloat16 CUDA cards**:
   `DITBackbone.forward` uses `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.
   On the test host's CUDA device (no bfloat16 support), the context manager
   constructor raises immediately. Also, `TimestepEmbedder.timestep_embedding()`
   hard-codes `dtype=torch.float32`, causing a dtype mismatch when model weights
   are bfloat16.

## Fix
All five fixes are in `tt_forge_models/mdlm/pytorch/loader.py`:

1. `_inject_flash_attn_stub()` — injects stub modules for `flash_attn`,
   `flash_attn.layers.rotary`, and `flash_attn.flash_attn_interface` into
   `sys.modules` with `ModuleSpec` set so `importlib.util.find_spec` succeeds.
   Stub implementations use `F.scaled_dot_product_attention` for attention and
   standard PyTorch cat/stack for RoPE.

2. `get_class_from_dynamic_module("modeling_mdlm.MDLM", ...)` — patches
   `MDLM.__init__` to call `self.post_init()` if `all_tied_weights_keys` is
   absent after the original `__init__` runs.

3. Set `tokenizer.pad_token = tokenizer.eos_token` wherever the GPT-2 tokenizer
   is initialized.

4. Replace `modulate_fused` in the remote module's `sys.modules` entry with a
   plain Python function `lambda x, shift, scale: x * (1 + scale) + shift` so
   TorchDynamo sees the correct (no-unsqueeze) definition directly.

5. Patch `DITBackbone.forward` to use `torch.amp.autocast(device_type=...,
   dtype=torch.bfloat16)` (device-safe). Patch `TimestepEmbedder.forward` to
   cast `t_freq` to the weight dtype before passing to the MLP.

RoPE cos/sin device mismatch fix: `_apply_rotary_emb_qkv_` moves cos/sin to
`qkv.device` before arithmetic (cos/sin can be CPU-cached from the CPU reference
run while qkv is on `xla:0`).

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    58.17s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mdlm/pytorch/loader.py` (new content added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 739436bc2e4bdbe5acd5c9b0d15a97e323cb5ebe |
| tt-forge-models | 4aa3aa313c9925203da77a560f5bb4ed66871ee3 |
