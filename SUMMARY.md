# Remediation Summary: kormo_vl-image_text_to_text-pytorch-KORMo-VL-KORMo-VL-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[kormo_vl/image_text_to_text/pytorch-KORMo-VL/KORMo-VL-single_device-inference]

## Result
FAIL — compiler generates a ~46 GiB CumSumOp input tensor during KORMo language model forward, exceeding total device DRAM (~32 GiB); root cause in lowering pipeline unknown

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-cumsum-shape-overflow-masked-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
2026-04-28 14:17:45.264 | critical |          Always | TT_FATAL: Out of Memory: Not enough space to allocate 49459232768 B DRAM buffer across 8 banks, where each bank needs to store 6182404096 B, but bank size is 4273390016 B (allocated: 3008150144 B, free: 1265239872 B, largest free block: 1247371200 B) (assert.hpp:104)
```

Call stack:
```
ttnn::operations::reduction::accumulation::common::preprocess_input_tensor
  ttnn::cumsum(tensor, dim, ...)
    tt::runtime::ttnn::operations::reduction::cumsum::run(CumSumOp, context)
      tt::runtime::ttnn::ProgramExecutor::execute()
        tt::runtime::submit(Device, Binary, ...)
          torch_xla.sync()
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Three loader bugs were fixed before reaching this failure:

1. **`KeyError: 'kormo'`** — `LlavaOnevisionConfig.__init__` calls `CONFIG_MAPPING["kormo"]` before any `auto_map` resolution; the KORMo text model classes must be pre-registered via `get_class_from_dynamic_module` from `KORMo-Team/KORMo-10B-sft`. Fixed in loader.

2. **L1 CB overflow (96 MB vs 1.5 MB limit)** — `LlavaOnevisionModel.get_placeholder_mask` constructs a 3D `[batch, seq_len, hidden_size]` boolean mask and calls `inputs_embeds[mask].numel()`, forcing a dynamic-size boolean gather onto TT device. Replaced with arithmetic: `n_image_tokens * inputs_embeds.shape[-1]`. Fixed in loader.

3. **Token count mismatch** — TT device int64 equality sum returns 2928 instead of 2929 for `(input_ids == 125041).sum()`. Forced computation to CPU. Fixed in loader.

After all loader fixes, the model advances to the KORMo language model forward pass (compiled graph for `LlavaOnevisionModel.forward`). The TTNN runtime executes a compiled flatbuffer containing a `CumSumOp` whose input tensor totals **49,459,232,768 bytes (≈46 GiB)**. This is physically impossible to allocate: the device has 8 GDDR banks of 3.98 GiB each (≈31.8 GiB total), and a single bank would need 5.76 GiB.

The tensor size (24,729,616,384 BF16 elements = 2948 × 4096 × 2048) does not correspond to any naturally-sized tensor in this model (seq_len=2948, hidden_size=4096, intermediate_size=16384). No Python-level `torch.cumsum` is called during the forward pass; the `CumSumOp` is generated internally by the XLA → StableHLO → TTIR lowering pipeline, most likely from the `inputs_embeds.masked_scatter(special_image_mask, image_features)` operation (shape [1, 2948, 4096]) which XLA lowers using cumsum-based scatter index computation. The inflated 46 GiB shape indicates a shape inference or tensor aliasing bug in the tt-mlir lowering.

## Fix
Not attempted (Tier B). The proposed investigation:
1. Dump the TTIR flatbuffer produced for the KORMo forward pass and locate the `CumSumOp`
2. Identify which stablehlo op was lowered to it and what its input shape was in the stablehlo IR
3. If the stablehlo has correct shapes but TTIR inflates them: fix the TTIR lowering pattern for stablehlo cumsum/reduce_window
4. If XLA generates stablehlo with already-inflated shapes: investigate torch_xla's lowering of `masked_scatter` or `iota`+cumsum patterns

Fix would live in `tt-mlir` (StableHLO → TTIR lowering patterns or TTNN execution memory planning).

## Tier B justification
Indicator: `internal-error-unknown-mechanism`

The CumSumOp input tensor size (46 GiB) has no plausible correspondence to any model tensor; the root cause — which specific lowering step inflates the shape — is unknown. Diagnosing it requires IR inspection across multiple lowering stages (stablehlo → TTIR → TTNN flatbuffer) and the fix location is unknown. This is diagnosis-first work for a compiler expert.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    341.44s (0:05:41) — OOM at ~176 s into compilation/execution
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/kormo_vl/image_text_to_text/pytorch/loader.py`
  - `_register_kormo_classes()`: register KORMo custom model classes before loading
  - `_patch_llava_placeholder_mask()`: replace device-side boolean gather with arithmetic equivalent; compute token counts on CPU
  - `_load_processor()`: use_fast=False for transformers 5.x compatibility
  - `load_inputs()`: pass dtype_override to pixel_values

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6e316c249e25601119b22e8632bbfbe6fd17cf25 |
| tt-forge-models | ba09d950ae00bda29286efe24d36bb98c15c2377 |
