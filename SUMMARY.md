# Remediation Summary: awportrait_z-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[awportrait_z/pytorch-Base-single_device-inference]

## Result
FAIL — tt-mlir cannot lower complex<f32> tensors produced by the Z-Image RoPE path (Tier B, new-infrastructure)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
stablehlo-complex-float-type-no-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original:
```
2026-04-23 23:32:16.471 | critical | Always | TT_FATAL: Chip 0 logical eth core (x=0,y=10) connects to a remote mmio device (assert.hpp:104)
```

Reproduced (loader bug first, then compiler bug after loader fix):
```
KeyError: 'layers.0.adaLN_modulation.0.alpha'
```
After loader fix:
```
loc("p2.9"): error: failed to legalize unresolved materialization from ('tensor<512x24x2xf32>') to ('tensor<512x24xcomplex<f32>>') that remained live after conversion
module_builder.cc:889 ERR| Failed to convert from SHLO to TTIR module
ValueError: Error code: 13
```

## Root cause
Two bugs were found in sequence:

**1. Loader bug (fixed):** AWPortrait-Z.safetensors uses `diffusion_model.*`-prefixed `lora_A`/`lora_B` keys with no alpha scaling entries. diffusers 0.37 routes `diffusion_model.*`-prefixed keys through `_convert_non_diffusers_z_image_lora_to_diffusers`, which unconditionally calls `state_dict.pop(alpha_key).item()`. Since no alpha keys exist, this raises `KeyError`. Convention when alpha is absent is `alpha = rank` → `scale = 1.0`.

**2. Compiler bug (Tier B):** The Z-Image transformer's `RopeEmbedder.precompute_freqs_cis` calls `torch.polar(...).to(torch.complex64)`, producing `complex<f32>` tensors. The `apply_rotary_emb` function uses `torch.view_as_complex` / `torch.view_as_real` around a complex multiplication. tt-mlir has no lowering for `complex<f32>` — the StableHLO→TTIR conversion fails with:
```
failed to legalize unresolved materialization from ('tensor<512x24x2xf32>') to ('tensor<512x24xcomplex<f32>>') that remained live after conversion
```
This is the same bug class as Canary NeMo STFT (XLAComplexFloatType Tier B).

The TT_FATAL `eth core connects to remote mmio device` messages in the original report are n300 hardware-topology warnings (chip 0's ethernet cores connect to chip 1), not the failure cause.

## Fix
**Loader fix committed:** `tt_forge_models` branch `remediation/awportrait_z-pytorch-Base-single_device-inference`, commit `942e9ba16a`.  
File: `awportrait_z/pytorch/loader.py`  
Change: Load LoRA safetensors directly via `hf_hub_download` + `safetensors.torch.load_file`, rename keys from `diffusion_model.X` → `transformer.X` (what the conversion does minus the absent alpha scaling), pass the pre-converted dict to `load_lora_weights` to bypass the broken `_convert_non_diffusers_z_image_lora_to_diffusers` code path.

**Compiler fix (proposed, not attempted):** Add `complex<f32>` type lowering support in tt-mlir's StableHLO→TTIR conversion pass. Specifically, `view_as_complex` / `view_as_real` operations and arithmetic on `complex<f32>` tensors would need to be decomposed into real-valued operations before lowering to TTIR. This is new infrastructure requiring changes across multiple tt-mlir passes.

## Tier B justification
**Indicator:** new-infrastructure  
Implementing `complex<f32>` type support in tt-mlir requires adding new op lowerings (complex arithmetic, `view_as_complex`, `view_as_real`) across multiple passes in the StableHLO→TTIR pipeline. This is not a scoped patch to an existing lowering.

## Verification
- pytest exit: FAIL
- Hardware:    n300
- Duration:    89.30s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/awportrait_z/pytorch/loader.py` (loader fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 942e9ba16a3b25efb2bc4a48859457ed37848f47 |
