# Remediation Summary: cambrian-cambrian_s-pytorch-S_3B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cambrian/cambrian_s/pytorch-S_3B-single_device-inference]

## Result
FAIL — TT silicon PCC=0.902 is below the 0.99 threshold; BF16 matmul accumulation precision floor in the 36-layer Qwen2.5-3B language model (same class as Qwen3 4B at 0.864, Gemma 7B at ~0.915)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original failure was `raise ValueError(` from `AutoModelForCausalLM.from_pretrained`
because the `cambrian_qwen` model_type was not registered in transformers 5.x
CONFIG_MAPPING.  After fixing the registration chain, a series of further
transformers 5.x incompatibilities were exposed and fixed (see Fix section below).
After all loader fixes the model runs on TT silicon but produces PCC=0.902 vs
the CPU golden reference (0.99 required).

## Root cause
Two distinct root causes:

**Loader layer (10 bugs fixed):**
The bundled Cambrian-S model code was written against transformers 4.x and
PyTorch/XLA patterns that are incompatible with transformers 5.2.0 and
TorchDynamo 2.7. Each bug caused a crash before the model could complete
a single forward pass.

**Compiler layer (Tier B):**
After all loader bugs were resolved, TT silicon produces PCC=0.902 on a 36-layer
Qwen2.5-3B language model.  Wormhole executes matmuls in BF16 with limited
accumulator precision; for large intermediate sizes the rounding error compounds
across layers.  The same root cause was previously identified for Qwen3 4B
(PCC=0.864) and Gemma 7B (~0.915), both filed as `ttmlir-f32-precision-not-preserved`.

## Fix

**Loader fixes (cambrian/cambrian_s/pytorch in tt-forge-models):**

`loader.py`:
- Register `CambrianQwenConfig` / `CambrianQwenForCausalLM` with `AutoConfig` /
  `AutoModelForCausalLM` so `from_pretrained` can resolve `cambrian_qwen` model_type.
- Add bundled `cambrian/` package directory to `sys.path` before importing.
- Reinit `SigLipVisionEmbeddings.position_ids` unconditionally after
  `from_pretrained`: the non-persistent buffer is materialized with uninitialized
  memory (not meta) by the meta-device code path in transformers 5.x.
- `load_inputs()` returns `images = [tensor[1,1,C,H,W]]` list + `image_sizes = [(w,h)]`
  as required by `prepare_inputs_labels_for_multimodal_for_generation`.

`cambrian_arch.py`:
- Wrap `vision_tower_aux_list` in `nn.ModuleList` so `from_pretrained` can navigate
  the module hierarchy and load checkpoint weights from the Cambrian checkpoint
  (plain Python list leaves modules unregistered; weights stay on meta device).
- Move `input_ids.tolist()` control flow to CPU to avoid `aten._local_scalar_dense`
  graph break on TT/XLA backends.

`llava_next_siglip_encoder.py`:
- `delay_load=True` branch now creates the `SigLipVisionModel` module structure
  (without downloading weights) so `from_pretrained` can load vision-tower weights
  that are stored inline in the Cambrian checkpoint.
- `SigLipVisionTower.device` property returns the actual device of the vision-tower
  parameters instead of unconditionally returning `xm.xla_device()`, which caused a
  device mismatch during the CPU golden reference run.
- `SigLipEncoder.forward()` no longer mutates `self._gradient_checkpointing_func`
  inside `forward()`.  The mutation caused TorchDynamo to treat `SigLipEncoder` as
  an `UnspecializedNNModuleVariable`, making `_getattr_static` unable to find
  `self.layers` (stored in `nn.Module._modules`, not `__dict__`).

`cambrian_qwen2.py`:
- Port `CambrianModel.forward()` to transformers 5.x API:
  - `get_usable_length` → `get_seq_length`
  - `self._attn_implementation` → `self.config._attn_implementation`
  - Pre-compute rotary `position_embeddings` before decoder loop and pass to each layer
  - `past_key_value` → `past_key_values` (kwarg rename in `Qwen2DecoderLayer`)
  - Handle `Qwen2DecoderLayer` returning a plain tensor (not a tuple) in 5.x
  - `next_decoder_cache` falls back to the in-place-updated `DynamicCache`
- Add `image_sizes` parameter to `CambrianQwenForCausalLM.forward()` and pass it to
  `prepare_inputs_labels_for_multimodal_for_generation`.

**Proposed compiler fix (Tier B, not attempted):**
Enable F32 accumulation for all matmuls on Wormhole hardware, or implement a
selective high-precision path for transformer MLP/attention projections in
`tt-mlir`.  This fix is cross-cutting (affects all WH BF16 models) and requires
coordinated changes across multiple files/passes in `tt-mlir`.

## Tier B justification
cross-cutting — the BF16 precision floor affects every model that runs large matmuls
through the Wormhole math engine; fixing it requires a cross-cutting precision policy
change in the `tt-mlir` lowering pipeline, not a scoped single-function change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~60s per run
- Tier A attempts: N/A

## Files changed
- `cambrian/cambrian_s/pytorch/loader.py` (complete rewrite)
- `cambrian/cambrian_s/pytorch/cambrian/` (new bundled package with all fixes):
  - `model/cambrian_arch.py`
  - `model/language_model/cambrian_qwen2.py`
  - `model/multimodal_encoder/llava_next_siglip_encoder.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fb0add656d05917403a6004f5d0ad575a7cd5c8d |
| tt-forge-models | ac27d7f1f9219f985b8dab480fb0c484e0cb20ef |
