# Remediation Summary: pe_av-pytorch-pe-av-large-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[pe_av/pytorch-pe-av-large-single_device-inference]

## Result
FAIL — PCC=0.0 from regular TTNN SDPA for non-tile-aligned KV sequence length after fixing decode-path crash

## Stack layer
tt-mlir, tt-metal

## Tier
B

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
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Triggered in `_align_video_hidden_state` during PE-AV audio-video encoder compilation via `torch_xla.sync()`.

After Tier A fix:
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.0. Required: pcc=0.99.

## Root cause

**First bug (Tier A, fixed):** `shouldUseDecode()` in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` checked only `querySeqLen == 1`, without checking whether the key sequence length was a multiple of 32. The `sdpa_decode` kernel enforces `k_chunk_size % 32 == 0`; `get_chunk_size(s)` returns the max power-of-2 divisor of `s`, so any `kSeqLen` not divisible by 32 produces `chunk_size < 32` → `TT_FATAL`. PE-AV's audio-video cross-attention has `qSeqLen=1` with `kSeqLen` not divisible by 32, hitting this path.

**Second bug (Tier B):** After the decode-path guard routes these ops to the regular TTNN SDPA path, the PCC drops to 0.0. The mechanism is unknown: non-tile-aligned KV lengths (sequences where `kSeqLen % 32 != 0`) trigger incorrect output from the regular SDPA kernel — the `use_padded_mask` program-factory path appears correct on inspection, but the kernel produces zeros. This is the same class of bug seen in Florence-2/DaViT encoder models.

## Fix

**Tier A fix (committed):** `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `shouldUseDecode()` now additionally requires `kSeqLen > 0 && (kSeqLen % 32 == 0)` before routing to the decode path. Branch: `remediation/pe_av-pytorch-pe-av-large-single_device-inference` in tt-mlir.

**Tier B (proposed):** The regular TTNN SDPA path produces PCC=0.0 for non-tile-aligned KV lengths. The fix likely lives in `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/` — the padding/mask logic for inputs where `kSeqLen % 32 != 0`. Root cause diagnosis required before a fix can be written.

## Tier B justification

**Indicator:** `internal-error-unknown-mechanism` — the regular SDPA kernel produces PCC=0.0 for non-tile-aligned KV inputs; the program factory's `use_padded_mask=True` logic looks correct on inspection, but the kernel output is wrong. The mechanism by which incorrect values propagate is unknown, making this a diagnosis-first problem.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    264.79s (0:04:24) for the final run
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `shouldUseDecode()` kSeqLen%32 guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 7e5f686c07a03cac7f516a4c6c8df54c874c5a51 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3a9d21f410cb5a1133c17f3cd072f477d9ad938a |
