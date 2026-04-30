# Remediation Summary: kimi_k25_eagle3-pytorch-Kimi_K25_Eagle3-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k25_eagle3/pytorch-Kimi_K25_Eagle3-single_device-inference]

## Result
SILICON_PASS — loader implemented Eagle3Speculator architecture; test passed on n150 in 60.18s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
eagle3-llmacausalLM-architecture-missing-in-hf-repo

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

Full transformer load report:
- UNEXPECTED keys: `midlayer.*`, `fc.weight`, `t2d`, `d2t` (EAGLE3-specific layers not in LlamaForCausalLM)
- MISSING keys: `model.embed_tokens.weight`, `model.layers.0.*` (standard Llama layers absent from checkpoint)
- MISMATCH: `lm_head.weight` checkpoint [32000, 7168] vs model [163840, 7168]

## Root cause
The `AQ-MedAI/Kimi-K25-eagle3` HuggingFace repo declares `architectures: ["LlamaForCausalLMEagle3"]` in config.json but ships no Python code defining that class. `AutoModelForCausalLM.from_pretrained` cannot resolve `LlamaForCausalLMEagle3` and silently falls back to `LlamaForCausalLM`. The checkpoint has EAGLE3-specific weight keys (`midlayer.*`, `fc.weight`, `d2t`, `t2d`) that don't exist in `LlamaForCausalLM`, and `lm_head.weight` has `draft_vocab_size=32000` rows vs the full Kimi `vocab_size=163840`. This mismatch triggers the `ignore_mismatched_sizes` exception.

The EAGLE3 draft model is part of the speculative decoding framework (SafeAILab/EAGLE3); its architecture class is not in transformers and is not available as a PyPI package.

## Fix
Implemented `Eagle3Speculator` architecture directly in the loader (`kimi_k25_eagle3/pytorch/loader.py` in tt_forge_models), matching the checkpoint's weight layout:

- `Eagle3Attention`: q/k/v project from concatenated `[embed_norm, hidden_norm]` (2×H=14336 → H=7168 via `num_heads*head_dim=8192`)
- `Eagle3DecoderLayer`: normalises embed and hidden independently, concatenates for attention, adds residual and MLP (reuses `transformers.models.llama.modeling_llama.LlamaMLP`)
- `Eagle3Speculator.fc`: `Linear(3×7168, 7168)` fuses 3 auxiliary hidden states from layers [2, 30, 58] of the verifier
- `Eagle3Speculator.lm_head`: `Linear(7168, 32000)` projects to draft vocabulary
- `d2t` / `t2d` buffers: vocabulary mapping arrays (int64 [32000] and bool [163840])
- `embed_tokens`: randomly-initialised `Embedding(163840, 7168)` — not in checkpoint; verifier embeddings are used at actual EAGLE3 inference time

Weights loaded directly from `model.safetensors` via `safetensors.torch.load_file` with `strict=False`, allowing only `embed_tokens.weight` to be missing.

`load_inputs` updated to return `(hidden_states [1,1,21504], input_ids [1,1])` matching the Eagle3Speculator forward signature; tokenizer dependency removed.

Changed files in `tt_forge_models` on branch `remediation/kimi_k25_eagle3-pytorch-Kimi_K25_Eagle3-single_device-inference`:
- `kimi_k25_eagle3/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    60.18s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/kimi_k25_eagle3/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | bb0f80fe916b79eeb57de6c93d8fff697eca6fe9 |
