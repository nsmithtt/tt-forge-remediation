# Remediation Summary: cv_vlm-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cv_vlm/pytorch-Base-single_device-inference]

## Result
FAIL — PCC=0.072 from regular TTNN SDPA on non-tile-aligned K/V sequences (ttnn-sdpa-nonaligned-kv-pcc-wrong)

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
Original crash: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`
Underlying: `TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2`

After loader fixes, a second crash in the same op: `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`

After all fixes: `AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.07217441660141126. Required: pcc=0.99.`

## Root cause
Four issues in a chain:

**1. Loader — transformers 5.x compatibility (four sub-issues):**

- `forced_bos_token_id` removed from `PretrainedConfig` in transformers 5.x. The remote `Florence2LanguageConfig.__init__` accesses `self.forced_bos_token_id` after `super().__init__()`, which no longer sets it → `AttributeError`.
- `PreTrainedModel.get_init_context` now always places model init in a `torch.device("meta")` context; `DaViT.__init__` calls `.item()` on a `torch.linspace()` tensor incompatible with meta tensors → `RuntimeError: Tensor.item() cannot be called on meta tensors`.
- `PreTrainedTokenizerBase.additional_special_tokens` renamed to `extra_special_tokens`; the remote processor code reads the old name.
- `CLIPImageProcessor` defaults to `use_fast=True` which produces non-square `pixel_values`; DaViT encoder requires square feature maps.

**2. tt-mlir — `shouldUseDecode` dispatches to SDPA decode for any q_seq_len == 1, even when k_seq_len % 32 ≠ 0:**

`sdpa_decode.cpp` computes `k_chunk_size` as the maximum power-of-2 divisor of `k_seq_len` and asserts it is a multiple of 32. With Florence-2's decoder having `q_seq_len=1` and self-attention `k_seq_len=1` (or small cross-attention sequence lengths), the calculated `k_chunk_size` is 2 → `TT_FATAL`.

**3. tt-metal — regular TTNN SDPA gives PCC ≈ 0.072 for non-tile-aligned sequences:**

After the `shouldUseDecode` guard routes all SDPA calls with non-tile-aligned k_seq_len to the regular SDPA path, the regular path computes incorrect results (PCC=0.072). The program factory pads K/V to the next tile boundary but produces wrong attention outputs. Mechanism unknown; `use_padded_mask=True` logic in the program factory does not appear to fix it.

## Fix
**Loader fix** (`tt_forge_models/cv_vlm/pytorch/loader.py`):
1. Set `PretrainedConfig.forced_bos_token_id = None` if absent before calling `from_pretrained`.
2. Temporarily remove the meta-device context from `PreTrainedModel.get_init_context` during model loading to allow DaViT's `.item()` calls.
3. Add `PreTrainedTokenizerBase.additional_special_tokens` property alias in `_load_processor`.
4. Pass `use_fast=False` to `AutoProcessor.from_pretrained`.

**tt-mlir Tier A fix** (`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`, `shouldUseDecode()`):
Added guard `kSeqLen > 0 && (kSeqLen % 32 == 0)` — SDPA decode is now only used when the key sequence length is a multiple of 32, preventing the `k_chunk_size < 32` kernel fatal.

**tt-metal Tier B FAIL — `ttnn-sdpa-nonaligned-kv-pcc-wrong`:**
Regular TTNN SDPA produces PCC=0.072 for Florence-2's non-tile-aligned sequences. Root cause unknown; requires investigation into the SDPA program factory's padding/mask logic for non-tile-aligned K/V.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism
The regular SDPA path returns PCC=0.072 for non-tile-aligned K/V inputs on Wormhole. The program factory's `use_padded_mask=True` path is present but produces wrong results. The mechanism for the precision error is unknown — it requires detailed kernel-level debugging of the SDPA compute shader, which is not scoped as a single-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    211.52s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/cv_vlm/pytorch/loader.py` — four transformers 5.x compatibility patches
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `shouldUseDecode` k_seq_len%32 guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | a03b89e5f308a3a363a7cfbe6af0535a4b1f60d8 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | af26a34475919258a2fc7857bcaf4f75089c8d63 |
