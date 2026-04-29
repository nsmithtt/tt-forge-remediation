# Remediation Summary: deepseek_ocr_gguf-causal_lm-pytorch-Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_ocr_gguf/causal_lm/pytorch-Q4_0-single_device-inference]

## Result
FAIL — `deepseek_vl_v2` GGUF architecture not supported by transformers; NexaAI GGUF uses non-standard metadata and stacked-expert tensor format incompatible with existing loader infrastructure

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-deepseek-vl-v2-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture deepseek_vl_v2 is not supported yet.
```

(The original CI failure `raise AttributeError(` would have come from `DeepseekVLForConditionalGeneration.generate()` — which raises `AttributeError("Not needed for DeepseekVL")` — had the model been loaded as `deepseek_vl` in an older transformers path. Our reproduction hits the stricter `ValueError` guard in transformers 5.2.0 before reaching inference.)

## Root cause
`NexaAI/DeepSeek-OCR-GGUF` (file `DeepSeek-OCR.Q4_0.gguf`) advertises `general.architecture = deepseek_vl_v2` in its header. This architecture string is absent from `transformers.integrations.ggml.GGUF_CONFIG_MAPPING`, so `load_gguf_checkpoint` raises `ValueError` at line 478 of `modeling_gguf_pytorch_utils.py`.

Beyond the missing registration, the GGUF was produced by a non-standard tool: all metadata fields use bare transformers-style names (`hidden_size`, `num_hidden_layers`, …) rather than the architecture-prefixed format expected by the GGUF parser (`deepseek_vl_v2.embedding_length`, etc.). As a result, adding `deepseek_vl_v2` to `GGUF_CONFIG_MAPPING` with a field mapping would still leave the config entirely empty (only `model_type` would be populated via the `general.architecture` mapping).

Additionally, the expert weights are stored as stacked 3-D tensors (`layers.N.mlp.gate_proj_experts.weight` with shape `[hidden, intermediate, num_experts]`) rather than per-expert 2-D tensors, requiring custom unstacking and transposition to fit any existing transformers MoE model class (e.g., `DeepseekV2Experts` which expects `[num_experts, 2*intermediate, hidden]` for its fused `gate_up_proj`).

## Fix
A complete fix requires new loader infrastructure:
1. Parse the non-standard bare-key GGUF metadata manually (using `gguf.GGUFReader`) to build a `DeepseekV2Config` (the text backbone is a 12-layer DeepSeek-V2 Lite MoE with `hidden_size=1280`, `n_routed_experts=64`, `n_shared_experts=2`, `num_experts_per_tok=6`, `first_k_dense_replace=1`).
2. Instantiate a `DeepseekV2ForCausalLM` from that config on meta-device.
3. Read tensors from the GGUF file and map names: add `model.` prefix, combine `gate_proj_experts` + `up_proj_experts` → `experts.gate_up_proj` (reshape and transpose), map `down_proj_experts` → `experts.down_proj`, handle `shared_experts.*` sub-module layout.
4. Load weights with `module.load_state_dict(state_dict, strict=True)`.

This fix lives entirely in `deepseek_ocr_gguf/causal_lm/pytorch/loader.py` but requires ~100 lines of custom tensor-name mapping logic and cannot be expressed as a simple GGUF arch registration.

## Tier B justification
**new-infrastructure**: The GGUF file uses a non-standard metadata format (bare key names) and stacked expert tensor layout for which no existing transformers GGUF path handles `deepseek_vl_v2`. Fixing it requires a custom tensor-loading pipeline inside the loader, not a scoped one or two-file change to the compiler stack.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    35.20s (to reproduce failure)
- Tier A attempts: N/A

## Files changed
None — no fix attempted (Tier B).

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
