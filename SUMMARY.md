# Remediation Summary: flux2_controlnet_fun-pytorch-ControlNet_Union-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux2_controlnet_fun/pytorch-ControlNet_Union-single_device-inference]

## Result
FAIL — WH BF16 matmul accumulation error causes PCC=0.9539, below required 0.99; Tier B compiler bug ttmlir-f32-precision-not-preserved

## Stack layer
loader, tt-metal

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
Original reported failure: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

After loader fixes, the test ran on silicon and produced:
```
AssertionError: PCC check failed: 0.9539 < required 0.99
```

## Root cause

**Loader bug 1 (fixed):** The original `loader.py` called `Flux2Transformer2DModel.from_single_file(f".../{repo_id}/resolve/main/{SAFETENSORS_FILE}", ...)`. The diffusers helper `_extract_repo_id_and_weights_name` only strips `/blob/main/` from URLs, not `/resolve/main/`, so it treated `resolve/main/FLUX.2-dev-Fun-Controlnet-Union.safetensors` as the filename path, causing a 404.

**Loader bug 2 (fixed):** The checkpoint (`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`) is a custom ControlNet with 76 keys under the `control_*` prefix — not a `Flux2Transformer2DModel`. Using `from_single_file` with the wrong class caused diffusers' model-type inference to fall back to `"v1"` (stable-diffusion-v1-5), leading to a config 404. The loader was rewritten to use `hf_hub_download` + `safetensors.torch.load_file` + a custom `FluxFunControlNetModel` nn.Module whose parameter keys exactly match the checkpoint.

**Compiler bug (unfixed, Tier B):** After the loader was fixed, TT silicon produces PCC=0.9539 against the CPU reference. CPU BF16 vs FP32 comparison gives PCC=0.9999, confirming the implementation is correct. The precision gap arises from Wormhole BF16 matmul accumulation error in wide reduction dimensions: `inner_dim=6144` for the attention projections and `ff.linear_out` reduction dim `18432 = 6144×3` for the gated FFN. This is the known `ttmlir-f32-precision-not-preserved` issue where intermediate F32 precision is not preserved through all StableHLO→TTIR→TTNN lowering passes. Fixing it requires cross-cutting changes across multiple passes.

## Fix

**Loader fixes** (tt_forge_models, branch `remediation/flux2_controlnet_fun-pytorch-ControlNet_Union-single_device-inference`):

1. `flux2_controlnet_fun/pytorch/loader.py` — removed `from_single_file` entirely; replaced with `hf_hub_download` + `safetensors.torch.load_file` + `FluxFunControlNetModel.load_state_dict(state_dict)`.

2. `flux2_controlnet_fun/pytorch/src/model.py` (new file) — custom `nn.Module` hierarchy:
   - `_FunControlNetFFN`: gated-GELU FFN with `linear_in` (expansion 6×) and `linear_out`, matching checkpoint keys.
   - `_FunControlNetAttn`: joint image+encoder attention with RMS-normed QK, matching `to_q/k/v`, `to_out.0`, `add_q/k_proj`, `to_add_out`, `norm_q/k/norm_added_q/k` keys.
   - `_FunControlNetBlock`: single block with optional `before_proj` (only block 0 has it), `after_proj`, `attn`, `ff`, `ff_context`.
   - `FluxFunControlNetModel`: top-level model with `control_img_in` + `control_transformer_blocks` (4 blocks).

3. `flux2_controlnet_fun/pytorch/src/__init__.py` (new file) — empty package init.

**Compiler fix:** Not attempted. This is the known `ttmlir-f32-precision-not-preserved` Tier B issue.

## Tier B justification

Which Tier B indicator applies: `cross-cutting`

Preserving F32 intermediate precision through the StableHLO→TTIR→TTNN lowering pipeline requires coordinated changes across multiple passes in tt-mlir (type inference, op lowering patterns, TTNN emission). It is not a single-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    93.27s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/flux2_controlnet_fun/pytorch/loader.py` (rewritten)
- `tt-xla/third_party/tt_forge_models/flux2_controlnet_fun/pytorch/src/__init__.py` (new)
- `tt-xla/third_party/tt_forge_models/flux2_controlnet_fun/pytorch/src/model.py` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c0ef1b9e437a50e83934d593b22d13eece93e8dc |
| tt-forge-models | 9a455606ed1799899c4dc602a00232bf30e778b1 |
