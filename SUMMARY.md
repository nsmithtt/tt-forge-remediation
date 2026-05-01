# Remediation Summary: darkmere_14b_v01_i1_gguf-causal_lm-pytorch-14B_v0.1_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[darkmere_14b_v01_i1_gguf/causal_lm/pytorch-14B_v0.1_i1_GGUF-single_device-inference]

## Result
FAIL — PCC 0.9831 < 0.99 on TT silicon; compiler-stack BF16 precision divergence (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-large-llm

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
raise NotImplementedError(
```
This was `NotImplementedError("Unknown gguf model_type: ministral3...")` from
`get_gguf_hf_weights_map` when transformers tried to build the tensor name map
for loading GGUF weights into the Ministral3ForCausalLM model.

After loader fixes, residual failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.9831063571986657. Required: pcc=0.99.
```

## Root cause
Two bugs, in two layers:

**Loader (tt-forge-models):** The GGUF file declares `general.architecture = mistral3`
(the text-only Ministral 3 architecture from gguf-py). Transformers 5.x knows `mistral3`
only as a VLM (`Mistral3ForConditionalGeneration`); the text model is `ministral3`
(`Ministral3ForCausalLM`). The GGUF loading pipeline failed because:
1. `mistral3` was absent from `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`.
2. `get_gguf_hf_weights_map` received `model_type='ministral3'` but gguf-py's
   `MODEL_ARCH_NAMES` uses `'mistral3'` as the key — the mismatch raised
   `NotImplementedError`.
3. ~29 other loaders globally patch `load_gguf_checkpoint` with the old 2-argument
   signature; transformers 5.x added `model_to_load=` as a third kwarg, causing
   `TypeError` when alphabetically-later loaders overwrote the darkmere patch.
4. `get_gguf_hf_weights_map` from onion008's patcher passes 5 positional args
   including `qual_name`; darkmere's import-time patch originally only accepted 4.

**Compiler (tt-mlir):** After all loader bugs were fixed, the model ran on TT silicon
but produced PCC 0.9831 vs CPU BF16. A CPU BF16 vs CPU FP32 measurement showed
PCC ≈ 1.0, confirming that BF16 precision loss from PyTorch BF16 arithmetic is
negligible for this model. The gap therefore reflects TT hardware computing BF16
operations differently from CPU PyTorch BF16 — consistent with the known
`ttmlir-bf16-precision-not-preserved` pattern where TT lowering passes do not
preserve the higher-precision accumulation that CPU BF16 matmul uses internally.
For a 40-layer 14B model, this accumulates to PCC 0.9831.

## Fix
**Loader fix (tt-forge-models, committed):**
- Register `mistral3` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`,
  and `GGUF_TO_FAST_CONVERTERS` at import time.
- Patch `get_gguf_hf_weights_map` at import time to remap `'ministral3'` → `'mistral3'`
  for gguf-py tensor-name map lookup. Both module-level and context-manager versions
  accept the full 5-argument signature `(hf_model, processor, model_type, num_layers, qual_name)`.
- Use a `_mistral3_load_context()` context manager in `load_model()` that temporarily
  wraps the current outermost `load_gguf_checkpoint` to strip `model_to_load` (new in
  transformers 5.x) before forwarding to the old-signature chain, and stores the model
  reference so `get_gguf_hf_weights_map` can recover it even when called with
  `hf_model=None`.
- Build `Ministral3Config` explicitly from GGUF metadata (via `_build_ministral3_config`)
  to bypass auto-config resolving `mistral3` → VLM; extract `rope_theta` from GGUF
  and wrap it in a `rope_parameters` dict.
- Guard `apply_chat_template` with `if self.tokenizer.chat_template is not None:`.

File changed: `darkmere_14b_v01_i1_gguf/causal_lm/pytorch/loader.py`

**Compiler fix (tt-mlir, NOT attempted — Tier B):** Would require preserving higher-precision
accumulation in BF16 matmuls across all TTIR lowering passes — a cross-cutting change.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

Fixing the BF16 precision gap requires ensuring TT hardware matmul accumulation matches
CPU PyTorch BF16 (which internally uses FP32 accumulation) across all TTIR lowering passes —
a coordinated change across multiple files and compilation stages with broad regression risk.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    596.80s (0:09:56) — silicon run with PCC check
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: darkmere_14b_v01_i1_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fa75342987ef2b152f1698dfd5f5dfd1b22565db |
| tt-forge-models | c3d6379fcbe9006a982b050e88029525e6915de8 |
