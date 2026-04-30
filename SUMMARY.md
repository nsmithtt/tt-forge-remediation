# Remediation Summary: igarin_qwen2_5_coder_7b_20260302_gguf-causal_lm-pytorch-7B_20260302_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[igarin_qwen2_5_coder_7b_20260302_gguf/causal_lm/pytorch-7B_20260302_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-patched-get-weights-map-positional-arg-count

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patch_transformers_qwen2vl_gguf.<locals>.patched_get_gguf_hf_weights_map() takes from 2 to 4 positional arguments but 5 were given

## Root cause
Cross-test contamination via module-level monkey-patches. Several loaders
(onion008, momix, noctrex, qwen_3_5_35b, qwen_3_5_claude_distilled) patch
`get_gguf_hf_weights_map` and call `orig_get_map` with 5 positional args:
`(hf_model, processor, model_type, num_layers, qual_name)`.

The `qwen_2_5_vl_72b_instruct_heretic_i1_gguf` loader also patches
`get_gguf_hf_weights_map` but uses `**kwargs` instead of an explicit
`qual_name` parameter, making its signature accept only 4 positional args.

When test collection imports both loaders and the qwen2vl patch ends up as
`orig_get_map` inside another loader's wrapper, the 5-positional-arg call
crashes with TypeError. The `igarin` loader itself is clean — it patches
nothing — but it calls `from_pretrained` which triggers `get_gguf_hf_weights_map`
via the contaminated chain.

## Fix
In `qwen_2_5_vl_72b_instruct_heretic_i1_gguf/image_to_text/pytorch/loader.py`,
added `qual_name=""` as an explicit 5th parameter to
`patched_get_gguf_hf_weights_map` and forwarded it by keyword to `orig_get_map`.
This makes the function signature compatible with callers that pass `qual_name`
positionally (matching the original transformers 5-arg signature for
`get_gguf_hf_weights_map`).

Repository: `tt-forge-models`
Branch: `remediation/igarin_qwen2_5_coder_7b_20260302_gguf-causal_lm-pytorch-7B_20260302_Q4_K_M_GGUF-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    440.27s (0:07:20)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: qwen_2_5_vl_72b_instruct_heretic_i1_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | a0574dffe65827bff7738d2b32bfdf4bacd19c17 |
| tt-forge-models | 4b4e081580e6d368682c5ac26e360388489e7638 |
