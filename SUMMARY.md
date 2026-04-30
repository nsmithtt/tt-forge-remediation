# Remediation Summary: internlm2-causal_lm-pytorch-Chat_7B_ExPO-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[internlm2/causal_lm/pytorch-Chat_7B_ExPO-single_device-inference]

## Result
FAIL — PCC=NaN on TT silicon after loader fix; compiler-stack gather lowering produces NaN in torch.compile path, cause unconfirmed

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
gather-2d-position-ids-nan-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

Reproduced failure (after loader fix):
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.
```

## Root cause

**Loader bug (fixed):** transformers 5.x normalizes `rope_scaling=null` from the model's `config.json` into `{"rope_type": "default", "rope_theta": 1000000.0}`. The InternLM2 custom modeling code (`trust_remote_code=True`) from `chujiezheng/internlm2-chat-7b-ExPO` was written for transformers 4.40.1. Its `_init_rope()` method checks `if self.config.rope_scaling is None` (evaluates False with the new dict), then falls through to `scaling_type = self.config.rope_scaling["type"]` — a `KeyError` on `"type"` because the key is `"rope_type"`.  This caused a crash before the model could be compiled. The fix resets `rope_scaling = None` when `rope_type == "default"` at config load time.

**Compiler-stack bug (unfixed):** After the loader fix the model runs end-to-end on TT silicon (2 perf iterations complete, benchmark times recorded) but the logit tensor contains NaN or is constant — PCC=NaN. The most probable cause is the `stablehlo.gather` lowering for rotary position embeddings: `cos[position_ids]` and `sin[position_ids]` where `cos`/`sin` are 2D tensors `[max_seq_len, head_dim//2]` and `position_ids` is 2D `[1, seq_len]`, producing 3D output `[1, seq_len, head_dim//2]`. In eager-XLA mode this manifested as an explicit MLIR verifier failure (`ttir.concat output dimension 0 does not match sum of inputs: 1 vs. 172`) from `StableHLOGatherToSliceRepeatConcatPattern`. In `torch.compile` mode the pattern likely rejects (the indices are not a `ttir.constant` in that path) and control falls to `StableHLOGatherToEmbeddingPattern` or another fallback; the lowering produces wrong values (NaN or garbage) that propagate through all 32 attention layers. The exact failing code path has not been confirmed by instrumentation.

## Fix

**Loader fix (committed and pushed):**
- Repo: `tt_forge_models` on branch `remediation/internlm2-causal_lm-pytorch-Chat_7B_ExPO-single_device-inference`
- File: `internlm2/causal_lm/pytorch/loader.py`
- Change: In `load_model()`, load `AutoConfig` before `AutoModelForCausalLM`, then set `config.rope_scaling = None` when `config.rope_scaling.get("rope_type") == "default"`. Pass the patched config to `from_pretrained`.

**Compiler-stack Tier A attempt (attempted and reverted):**
- Repo: `tt-mlir`
- File: `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- Change tried: Added rank guard in `StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite` that returns `notifyMatchFailure` when the output rank differs from the input rank (indicating batch dimensions in the start indices). This correctly prevented the verifier failure in eager-XLA mode but did not affect the `torch.compile` path and did not fix PCC=NaN.
- Reverted per skill rules (one Tier A attempt, test still fails).

**Proposed fix for the remaining bug:**
Investigate what lowering `stablehlo.gather` uses in `torch.compile` mode when the output rank is one greater than the input rank (2D indices into a 2D table → 3D output). Either (a) extend `StableHLOGatherToEmbeddingPattern` to handle this 3D case, or (b) add a new pattern that reshapes the 2D index to 1D, calls the 1D embedding lookup, then reshapes the output back to 3D. The fix is localized to `StableHLOToTTIRPatterns.cpp`, but the root-cause path (which pattern is active in `torch.compile` mode) must be confirmed by logging before the fix can be written.

## Tier B justification

`internal-error-unknown-mechanism` — The exact code path that produces NaN in `torch.compile` mode has not been confirmed. Instrumentation (e.g. logging which gather pattern fires, or printing the intermediate cos/sin tensors on device) is required before a correct targeted fix can be written. The Tier A attempt was based on a plausible hypothesis that turned out not to affect the failing execution path.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    105.16s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/internlm2/causal_lm/pytorch/loader.py` (loader fix, committed)
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (rank guard, attempted and reverted)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4ce526224acc20581a67f91f17c2bce5d006ce34 |
| tt-forge-models | 5a4478ae3efa381b59ad358dc84518e891aacb0f |
