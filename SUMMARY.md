# Remediation Summary: mradermacher_nanovel_vl_i1_gguf/image_to_text/pytorch-nanovel_vl_i1_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_nanovel_vl_i1_gguf/image_to_text/pytorch-nanovel_vl_i1_gguf-single_device-inference]

## Result
FAIL — terminal conv3d-patch-embed-l1-overflow Tier B in Qwen3VL vision encoder

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
conv3d-patch-embed-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original (pre-fix): ValueError: GGUF model with architecture qwen3vl is not supported yet.

Terminal (post-fix): RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

The INTERNAL:13 occurs at `torch_xla._XLAC._xla_step_marker` called from
`torch_xla.sync()` inside `dynamo_bridge.extract_compiled_graph_helper`, during
compilation of the Qwen3VL vision encoder's `Qwen3VLVisionModel.forward` containing
the Conv3d patch embedding layer.

## Root cause
**Loader bug (fixed):** The GGUF file stores `general.architecture = "qwen3vl"` (no
underscore), which is absent from transformers' `GGUF_SUPPORTED_ARCHITECTURES`. This
raises `ValueError: GGUF model with architecture qwen3vl is not supported yet.` before
any weights are loaded.  Additionally, the 26 narrow-signature `load_gguf_checkpoint`
monkey-patches installed by other GGUF loaders in the same session reject the
`model_to_load` kwarg introduced in transformers 5.2.0, causing TypeError.  The GGUF
ships no `vision_config`, so the default `out_hidden_size=3584` disagrees with the
model's `hidden_size`.  Several `.tolist()` calls on device tensors hang on TT silicon.

**Terminal compiler bug (unfixed):** `Qwen3VLVisionPatchEmbed` uses
`Conv3d(C_in=3, C_out=~1152, kernel=[2,16,16])`.  The weights CB for this layer is
~2.1 MB, exceeding the 1.5 MB per-core L1 budget.  The tt-metal Conv3d sharded
program factory has no guard for this case, so the compiled program raises INTERNAL:13
(OOM) at runtime.

## Fix
**Loader fixes** in
`tt_forge_models/mradermacher_nanovel_vl_i1_gguf/image_to_text/pytorch/loader.py`
on remediation branch `remediation/mradermacher_nanovel_vl_i1_gguf-image_to_text-pytorch-nanovel_vl_i1_gguf-single_device-inference`
of tt-forge-models (commit 558d885167):

1. **`requirements.txt`** — add `gguf>=0.10.0` so the GGUF reader is available.

2. **`_register_qwen3vl_gguf_support()`** (module-level) — appends `"qwen3vl"` to
   `GGUF_SUPPORTED_ARCHITECTURES` and inserts the field-name mapping into
   `GGUF_TO_TRANSFORMERS_MAPPING["config"]`; also propagates the qwen3 tokenizer
   converter to the `qwen3vl` and `qwen3_vl` keys.

3. **Wide-sig `_nanovel_load_gguf` wrapper** — installs a `(*args, **kwargs)`-accepting
   replacement for `load_gguf_checkpoint` during `from_pretrained`, avoiding the
   narrow-sig TypeError from other loaders' patches.  For config pass: delegates to the
   chain with `return_tensors=False` and translates the flat `qwen3vl` dict into a
   nested `Qwen3VLConfig` structure (`text_config` sub-dict, `vision_config.out_hidden_size`).
   For tensor pass (`model_to_load` present): loads directly via `GGUFReader` + `dequantize`
   using a hard-coded qwen3vl→HF parameter name mapping, bypassing the chain entirely.
   The wrapper is installed/restored in a try/finally around `from_pretrained`.

4. **`_patch_qwen3vl_for_tt_device(model=)`** — patches four methods after `model.eval()`
   to move metadata tensors (grid_thw, input_ids, attention_mask) to CPU before `.tolist()`
   control flow: `rot_pos_emb`, `get_rope_index`, `get_image_features`,
   `fast_pos_embed_interpolate`.  `fast_pos_embed_interpolate` is fully reimplemented on
   CPU (using a captured `pos_embed.weight` snapshot) with `xm.send_cpu_data_to_device`
   to return results to TT device without premature sync.

5. **Pixel limits** — `min_pixels=56*56`, `max_pixels=13*28*1280` on the image processor
   to cap patch count within hardware memory budget.

**Terminal bug (not fixed):** `conv3d-patch-embed-l1-overflow` in tt-metal Conv3d sharded
kernel — no fix attempted; Tier B.

## Tier B justification
cross-repo — fixing Conv3d L1 overflow requires coordinated changes across tt-metal
(sharded Conv3d program factory must guard c_in_block against L1 budget) and potentially
tt-mlir (to lower or shard Conv3d differently). Multiple files across two repos.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    740.84s (0:12:20)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mradermacher_nanovel_vl_i1_gguf/image_to_text/pytorch/loader.py`
- `tt_forge_models/mradermacher_nanovel_vl_i1_gguf/image_to_text/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 558d885167ea83a5053fafbf7aa631e5c835974f |
