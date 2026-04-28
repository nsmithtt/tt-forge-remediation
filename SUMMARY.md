# Remediation Summary: florence_2-image_captioning-pytorch-Base_Ft-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[florence_2/image_captioning/pytorch-Base_Ft-single_device-inference]

## Result
FAIL — regular TTNN SDPA gives PCC=0.087 for Florence-2 with non-tile-aligned sequence lengths (encoder k=580, decoder cross k=580, decoder self k=1)

## Stack layer
loader, tt-mlir, tt-metal

## Tier
A

## Bug fingerprint
ttnn-sdpa-nonaligned-kv-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Verbose logs (before Tier A fix) showed:
```
TT_FATAL @ sdpa_decode.cpp:63: k_chunk_size % 32 == 0
"Chunk size must be multiple of 32, k_chunk_size is: 2"
```

After the Tier A fix (shouldUseDecode guard):
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
    Calculated: pcc=0.08741913206131083. Required: pcc=0.99.
FAILED tests/runner/test_models.py::... in 186.60s (0:03:06)
```

## Root cause
Three bugs were found:

**Bug 1 (loader): Florence-2 transformers 5.x incompatibility.**
`Florence2LanguageConfig` no longer has `forced_bos_token_id` in transformers
5.x (moved to GenerationConfig). The custom remote-code `__init__` raises
`AttributeError`. DaViT's vision encoder also calls `torch.linspace().item()`
during model construction on meta device, raising `RuntimeError`.

**Bug 2 (tt-mlir, Tier A): SDPA decode selected for k_len % 32 ≠ 0.**
`shouldUseDecode()` in `ScaledDotProductAttentionOpConversionPattern` only
checked `q_len == 1` before routing to decode mode. For Florence-2, the
decoder self-attention has q=1, k=1 (one token) and the decoder cross-attention
has q=1, k=580. tt-metal's `get_chunk_size(s)` returns the largest power-of-2
divisor of `s`, capped at 512. For k=1: chunk_size=2; for k=580: chunk_size=4.
The decode kernel asserts `k_chunk_size % 32 == 0`, causing the crash.

Fixed in tt-mlir: guard decode mode to require `kSeqLen > 0 && kSeqLen % 32 == 0`.

**Bug 3 (tt-metal, second bug, Tier B): Regular TTNN SDPA gives PCC=0.087.**
After Bug 2 is fixed, all Florence-2 SDPA calls route to the regular (non-decode)
TTNN SDPA kernel. Florence-2 uses bfloat16 and the composite SDPA path.
The sequence lengths are never tile-aligned: encoder k=580, decoder self k=1,
decoder cross k=580. The program factory sets `use_padded_mask=True` for
non-tile-aligned non-causal inputs (generates a -inf mask for padded K
positions). Despite this, the final output PCC against CPU reference is
0.087 — essentially uncorrelated. Root cause is unknown: the padding mask
logic appears correct, the streaming path is disabled (Sq_chunk_t=1 ≤
qk_out_subblock_h), and scale is correctly defaulted to 1/sqrt(head_dim).
The "Sq_chunk_t==1 with K padding has L1 acc write-back issues after
cb_push_back_hold_wr_ptr" comment in sdpa_program_factory.cpp refers to the
streaming path (disabled here), not the legacy non-streaming path in use.

## Fix

**Loader fix** (committed): `_florence2_compat_load()` in
`tt-xla/third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py`
patches `PretrainedConfig.__init__` to restore `forced_bos_token_id=None` and
patches `torch.linspace` to force CPU during DaViT construction.
Branch: `remediation/florence_2-image_captioning-pytorch-Base_Ft-single_device-inference`
in tt_forge_models (commit `8fef2b4f3d`).

**Tier A compiler fix** (committed): `shouldUseDecode()` in
`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — added kSeqLen % 32 == 0
guard.
Branch: `remediation/florence_2-image_captioning-pytorch-Base_Ft-single_device-inference`
in tt-mlir (commit `228097151`).

**Bug 3 proposed fix**: Investigate why the regular TTNN SDPA kernel gives
PCC=0.087 for non-tile-aligned K/V (580→608, 1→32) with `use_padded_mask=True`.
Candidates: SDPA reader kernel's padded-mask generation for Sq_chunk_t=1
(legacy non-streaming path), accumulation correctness for the decoder
cross-attention case (Sq=1 padded to 32, Sk=580 padded to 608). Likely lives
in `sdpa_program_factory.cpp` or the compute kernel `sdpa.cpp` in tt-metal.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    186.60s (0:03:06)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 2280971511f052b96bc41cf7f25d9e28aeeaf619 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8fef2b4f3d6a9b374fdafd2a6f34ca1151095c4d |
