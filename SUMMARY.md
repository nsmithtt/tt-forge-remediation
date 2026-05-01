# Remediation Summary: kyutai_tts-pytorch-TTS_1_6B_en_fr-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kyutai_tts/pytorch-TTS 1.6B en_fr-single_device-inference]

## Result
SILICON_PASS — all 32 codebook streams achieve PCC > 0.99 (overall PCC 0.9990)

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
tt-bfloat16-int64-roll-precision-loss

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original error: `ERROR: file or directory not found: 1.6B` (pytest argument parsing — spaces in test name require quoting).

After quoting, the model ran but produced `pcc=0.4613`. Per-codebook analysis showed:
- CB 0 (depformer codebook 0, delay=0): PCC=0.9999 ✓
- CB 1 (depformer codebook 1, delay=2): PCC=0.9974 ✓
- CBs 2-31 (depformer codebooks 2-31, delay=2): PCC=0.07–0.39 ✗

Two separate root causes were found and fixed.

## Root cause

**Bug 1 — `_undelay_sequence` NaN propagation (loader):**
`moshi.models.lm._undelay_sequence` fills delay-masked logit positions
with `float('NaN')` via in-place assignment `line[:, -delay:] = NaN`.
TT hardware does not preserve bfloat16 NaN; the in-place fill produces
`max_bfloat16 (~3.39e38)` or Inf in the compiled graph, corrupting the
output logits for all codebooks with delay > 0.

**Bug 2 — `_delay_sequence` int64 roll precision loss (loader):**
`moshi.models.lm._delay_sequence` applies `tensor[:, k].roll(delay,
dims=1)` to shift code indices (int64, range 0–2047) before feeding
them to the depformer embeddings. TT hardware operates in bfloat16
internally; `roll()` on int64 tensors converts indices to bfloat16,
which can only represent integers exactly up to 2^8=256. Codes in the
range 257–2047 are rounded (e.g. 279→280, 1860→1856), causing wrong
embedding lookups in the depformer. All depformer codebooks with
delay > 0 (CBs 2-31 of the logits) received wrong embedding inputs
and produced PCC=0.07–0.39. Codebooks with delay=0 were unaffected
because `roll(0)` is a no-op and no integer arithmetic occurred on TT.

Isolation tests confirmed: the depformer transformer itself and
`forward_depformer_training` with CPU-pre-computed inputs both achieve
PCC > 0.997 on TT. The precision loss was introduced exclusively in
the `_delay_sequence` call inside the compiled full-model graph.

## Fix

Both fixes are in `kyutai_tts/pytorch/loader.py` in tt_forge_models,
on branch `remediation/kyutai_tts-pytorch-TTS_1_6B_en_fr-single_device-inference`.

**Fix 1** (`_undelay_sequence` patch, commit 08ddf0dd82):
Patch `moshi.models.lm._undelay_sequence` to replace in-place NaN
fill with a functional `torch.cat([rolled[:T-delay], zeros_like[:delay]])`.
This produces 0.0 at delay-masked positions instead of NaN, keeping all
output values finite and allowing `out.logits` to be returned directly.

**Fix 2** (`_delay_sequence` patch, commit 89adbd0c13):
Patch `moshi.models.lm._delay_sequence` to replace `roll(delay) +
in-place fill` with functional `torch.cat([pad.expand(B,delay),
tensor[:, k, :T-delay]])`. This avoids any integer arithmetic on TT
hardware; only slicing is performed on the int64 code tensors, which
does not trigger bfloat16 conversion.

Additionally, `tt-xla` has a commit (95a1174ce) adding a
`convert_sdpa_bool_masks` FX pass that converts boolean SDPA attention
masks to additive float (0.0/-inf) masks before compilation. This was
needed because the depformer's causal attention mask is constructed as a
bool tensor at runtime, and TT's SDPA requires additive float masks.

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 204.69s (0:03:24)
- Tier A attempts: N/A

## Files changed
- `kyutai_tts/pytorch/loader.py` in tt_forge_models (two patches: `_undelay_sequence` and `_delay_sequence`)
- `python_package/tt_torch/backend/passes.py` in tt-xla (`convert_sdpa_bool_masks` FX pass)
- `python_package/tt_torch/backend/backend.py` in tt-xla (wire up `convert_sdpa_bool_masks` in pipeline)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 95a1174ce |
| tt-forge-models | 89adbd0c13 |
