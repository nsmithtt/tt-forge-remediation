# Remediation Summary: grape_mini_instruct_i1_gguf-image_to_text-pytorch-GRAPE_MINI_INSTRUCT_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[grape_mini_instruct_i1_gguf/image_to_text/pytorch-GRAPE_MINI_INSTRUCT_I1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Conv3d L1 overflow (Tier A) fixed; second bug revealed: PCC = -0.54 (Tier B, unknown root cause)

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
conv3d-small-cin-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_THROW @ .../tt_metal/third_party/umd/device/chip_helpers/silicon_sysmem_manager.cpp:326: tt::exception
```
Reproduced locally as:
```
INTERNAL: Error code: 13
```
during `torch_xla.sync()` while compiling the Conv3d patch embedding.

## Root cause

**Loader bugs (fixed):**
1. `general.architecture = "qwen3vl"` was absent from `GGUF_CONFIG_MAPPING` / `GGUF_SUPPORTED_ARCHITECTURES` in transformers, causing the GGUF load to fail immediately.
2. Qwen3VL's `fast_pos_embed_interpolate` calls `.tolist()` on device tensors (control flow), which triggers premature PJRT device-to-host transfers. Patched with `@torch.compiler.disable`.
3. `get_rope_index` / `get_placeholder_mask` call `.item()` / `.tolist()` on XLA tensors. Patched with `@torch.compiler.disable`.
4. `_deepstack_process` uses boolean advanced indexing (`hidden_states[mask, :]`) which has a data-dependent output shape incompatible with the TT static compiler. Replaced with `masked_scatter`.

**Compiler bug — Tier A — fixed:**
The Qwen3VL patch embedding uses `nn.Conv3d(in_channels=3, out_channels=1152, kernel_size=[2,16,16])`. In `TTIRToTTNN.cpp`, the Conv3d lowering unconditionally set `c_in_block = TILE_WIDTH = 32`, producing:
- vol2col_tiled CB: `c_in_block * kernel_elements * tile_bytes = 32 * 512 * 64 = 1,048,576 B`  (1 MB)
- weight_tiled CB: `c_in_block * c_out_block * tile_bytes = 32 * 32 * 64 = 65,536 B` (64 KB)
- Total ≈ 1.1 MB per CB × 2 double-buffer = 2.2 MB, exceeding the 1.5 MB L1 limit.

Fix: compute `maxCInBlock = max(1, MAX_CB_TILES * TILE_WIDTH / kernelElements)` where `MAX_CB_TILES = 256` (512 KB budget), then halve `cInBlock` until it satisfies both the L1 budget and the alignment constraint.  For this kernel `kernelElements = 2*16*16 = 512`, giving `maxCInBlock = 256*32/512 = 16`, so `c_in_block` drops from 32 → 16.

**Second compiler bug — Tier B — not fixed:**
After the Conv3d fix, the model compiles and runs end-to-end (~458 s) but produces PCC = -0.5434 against the CPU reference. This severe anti-correlation is not a BF16 precision floor (which would give PCC > 0.75) and is not consistent with the SDPA non-aligned-kv bug (which gives PCC ≈ 0.087). The root cause is unknown; the sign reversal pattern suggests a possible negation or misrouted tensor somewhere in the lowered graph. No obvious single-function hypothesis could be formed → Tier B.

## Fix

**Loader — `tt_forge_models` remediation branch** (`remediation/grape_mini_instruct_i1_gguf-image_to_text-pytorch-GRAPE_MINI_INSTRUCT_I1_Q4_K_M_GGUF-single_device-inference`):
- `grape_mini_instruct_i1_gguf/image_to_text/pytorch/loader.py`: register `qwen3vl` in `GGUF_CONFIG_MAPPING`; patch `fast_pos_embed_interpolate`, `get_rope_index`, `get_placeholder_mask` with `@torch.compiler.disable`; replace boolean-index `hidden_states[mask, :]` with `masked_scatter`
- `grape_mini_instruct_i1_gguf/image_to_text/pytorch/requirements.txt`: add `gguf>=0.10.0`

**Compiler — `tt-mlir` remediation branch** (`remediation/grape_mini_instruct_i1_gguf-image_to_text-pytorch-GRAPE_MINI_INSTRUCT_I1_Q4_K_M_GGUF-single_device-inference`):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: compute L1-safe `cInBlock` for Conv3d lowering using `MAX_CB_TILES = 256`

**Proposed fix for second bug:** Bisect the Qwen3VL forward pass to isolate which sub-graph produces the wrong sign. Candidates include the Conv3d multi-block accumulation path (unlikely — zero-padded second block contributes nothing), SDPA (would give near-zero PCC not negative), or a normalization op with a sign error. This requires interactive debugging with intermediate tensor captures and is beyond the scope of a single Tier A attempt.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

The second compiler bug produces PCC = -0.54 (severe anti-correlation). No single named function or formula change was identified as the root cause. The sign-reversal pattern is unusual enough that diagnosis (capturing intermediate activations, bisecting the lowered graph) must precede any fix attempt.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    458s (second run, after Conv3d fix)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/grape_mini_instruct_i1_gguf/image_to_text/pytorch/loader.py`
- `tt_forge_models/grape_mini_instruct_i1_gguf/image_to_text/pytorch/requirements.txt`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 441f850a1e7c8e67630537397e7f231847c08e25 |
| tt-xla          | adbe5f8132d88f252ee9fdd099faa288d9d3c600 |
| tt-forge-models | 9a19d013eead203dfcd97ca26732ed768b236f20 |
