# Remediation Summary: l3_grand_story_darkness_moe_4x8_24_9b_e32_gguf-causal_lm-pytorch-L3_GRAND_STORY_DARKNESS_MOE_4X8_24_9B_E32_Q4_K_M_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[l3_grand_story_darkness_moe_4x8_24_9b_e32_gguf/causal_lm/pytorch-L3_GRAND_STORY_DARKNESS_MOE_4X8_24_9B_E32_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — deterministic PCC=0.7245965198932484 between TT device logits and CPU reference logits; root cause not identified to a single op

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
llama-gguf-logits-pcc-systematic-divergence

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7245965198932484. Required: pcc=0.95.

## Root cause

The model DavidAU/L3-Grand-Story-Darkness-MOE-4X8-24.9B-e32-GGUF is a MoE-extended LLaMA-3 model. Its GGUF file uses the `llama` architecture tag, so `AutoModelForCausalLM.from_pretrained` loads it as a standard `LlamaForCausalLM`. The transformers GGUF loader (`GGUF_CONFIG_MAPPING['llama']`) does not map `expert_count` / `expert_used_count`, so `LlamaConfig` has no `num_local_experts` field. Consequently the expert weight tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) have no matching HF parameter and are silently skipped during loading.

The resulting model has randomly-initialised MLP projections. Both the CPU golden run and the TT device run use the same randomly-initialised model, so the GGUF loading issue alone cannot explain PCC divergence — it only means both sides compute the wrong model together. The persistent, deterministic PCC=0.72 (identical across two independent test runs) therefore reflects a systematic error in how the TT compiler stack computes the forward pass of this LlamaForCausalLM architecture relative to CPU.

The specific failing op or pass was not identified within the allotted diagnostic budget.

As a correct loader improvement (following the pattern of 107 other causal_lm loaders), `model.config.use_cache = False` was added to remove KV-cache tensors from the PCC comparison surface. This did not change the test outcome: the logits themselves already had PCC=0.72.

## Fix
No compiler-stack fix attempted (Tier B). Loader improvement committed:
- `tt-forge-models` `l3_grand_story_darkness_moe_4x8_24_9b_e32_gguf/causal_lm/pytorch/loader.py`: added `model.config.use_cache = False` after `from_pretrained` (commit `f16b7e28bd` on `remediation/l3_grand_story_darkness_moe_4x8_24_9b_e32_gguf-causal_lm-pytorch-L3_GRAND_STORY_DARKNESS_MOE_4X8_24_9B_E32_Q4_K_M_GGUF-single_device-inference`).

## Tier B justification
`internal-error-unknown-mechanism` — The PCC=0.72 is deterministic and systematic (same value on two independent runs, only logits compared after use_cache=False fix), indicating a real compiler bug, but identifying the specific divergent op requires instrumented layer-by-layer diagnosis that is beyond a single-attempt Tier A fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    812.70s (0:13:32)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/l3_grand_story_darkness_moe_4x8_24_9b_e32_gguf/causal_lm/pytorch/loader.py` — added `model.config.use_cache = False`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e944d8cebdf363edc9be85955aa73142ba753af3 |
| tt-forge-models | f16b7e28bd91c28410422a4c59f27f3cbece35dd |
