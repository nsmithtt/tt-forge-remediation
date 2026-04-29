# Remediation Summary: florence_2-image_captioning-pytorch-SD3-Captioner-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[florence_2/image_captioning/pytorch-SD3-Captioner-single_device-inference]

## Result
FAIL — regular SDPA gives PCC=0.12 for non-tile-aligned k_seq_len=1 (BART decoder self-attention after SDPA decode guard fix)

## Stack layer
loader, tt-mlir

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
Original failure chain:
1. `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'` (loader bug)
2. `TypeError: Florence2Seq2SeqLMOutput.__init__() got an unexpected keyword argument 'loss'` (loader bug)
3. `TT_FATAL @ tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:52: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2` (compiler bug, Tier A)
4. After Tier A fix: `PCC comparison failed. Calculated: pcc=0.12008863481244995. Required: pcc=0.99` (compiler bug, Tier B)

## Root cause
Three independent bugs at two layers:

**Loader (bugs 1 & 2):**
1. transformers 5.x removed `forced_bos_token_id` from `PretrainedConfig` and moved it to `GenerationConfig`. Florence-2 SD3-Captioner's remote `configuration_florence2.py` checks `self.forced_bos_token_id` after `super().__init__()`, raising `AttributeError`.
2. The remote `modeling_florence2.py` returns `loss`, `logits`, and `image_hidden_states` from `forward()` as kwargs to `Florence2Seq2SeqLMOutput(...)`, but the remote `@dataclass` definition only has 8 fields (missing those three). transformers 5.x strict `@dataclass` `ModelOutput` rejects unknown constructor kwargs.

**Compiler — tt-mlir (bug 3, Tier A):**
`shouldUseDecode()` in `TTIRToTTNN.cpp` selects SDPA decode when `query_seq_len == 1`, but does not verify that `k_seq_len % 32 == 0`. For Florence-2's BART decoder with a single-token decode step, `k_seq_len=1`. TTNN's `get_chunk_size(1)` returns 2 (the inner loop never advances), triggering `TT_FATAL: Chunk size must be multiple of 32`.

**Compiler — tt-mlir/tt-metal (bug 4, Tier B):**
After the decode guard fix, regular SDPA is selected for `k_seq_len=1`. TTNN regular SDPA produces incorrect results (PCC≈0.12) for non-tile-aligned K sequence lengths. This is the known `ttnn-sdpa-nonaligned-kv-pcc-wrong` bug.

## Fix
**Loader fixes** in `tt-xla/third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py` on branch `remediation/florence_2-image_captioning-pytorch-SD3-Captioner-single_device-inference` in tt_forge_models:

- Commit `2acd0080ac`: Added `_florence2_compat_load()` that wraps `PretrainedConfig.__init__` to inject `forced_bos_token_id=None` after init (for bug 1), and patches `torch.linspace` to force CPU to avoid DaViT meta-device `.item()` errors during model construction.
- Commit `8d04f2d40f`: After loading the model, locates the remote `Florence2Seq2SeqLMOutput` class via `sys.modules[model.__class__.__module__]` and replaces it with a complete `@dataclass` including all fields returned by `forward()`: `loss`, `logits`, `last_hidden_state`, `past_key_values`, `decoder_hidden_states`, `decoder_attentions`, `cross_attentions`, `encoder_last_hidden_state`, `encoder_hidden_states`, `encoder_attentions`, `image_hidden_states`.

**Compiler fix** in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` on branch `remediation/florence_2-image_captioning-pytorch-SD3-Captioner-single_device-inference` in tt-mlir:

- Commit `c9414b3f9`: Added `k_seq_len % 32 == 0` guard to `shouldUseDecode()`. If the key sequence length is not a multiple of 32, decode mode is not selected and the op falls through to regular SDPA.

**Proposed fix for Tier B bug (not implemented):**
The `ttnn-sdpa-nonaligned-kv-pcc-wrong` bug requires either padding K/V tensors to the next tile boundary (32) before SDPA and unpadding afterwards, or fixing the TTNN SDPA kernel to correctly handle non-tile-aligned inputs. This lives in tt-mlir's SDPA lowering or tt-metal's SDPA kernel implementation.

## Tier B justification
**cross-cutting**: Fixing regular SDPA for non-tile-aligned K/V requires either: (1) padding infrastructure in the SDPA lowering pass that inserts pad/unpad ops around the kernel call, or (2) fixing the TTNN SDPA kernel to handle non-multiple-of-32 sequence lengths. Either approach touches multiple files across tt-mlir and/or tt-metal and has broad potential for regression in other models using SDPA.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    N/A (failed at PCC check after ~10 minutes)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py` (loader fixes, 2 commits)
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (shouldUseDecode guard, 1 commit)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | c9414b3f96c3ada89bec08597559827954bbe961 |
| tt-xla          | 042c16ab7df90c0074d1f9e44fe60920e12ed9ce |
| tt-forge-models | 8d04f2d40f |
