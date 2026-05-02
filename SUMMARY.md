# Remediation Summary: nemotron_embed_vl_pytorch-Embed_VL_1B_V2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nemotron/nemotron_embed_vl/pytorch-Embed_VL_1B_V2-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-model-output-all-none-empty-pytree

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: max() iterable argument is empty
at tests/infra/evaluators/torch_comparison_evaluator.py:114

## Root cause
The `nvidia/llama-nemotron-embed-vl-1b-v2` model's `forward()` returns a
`CausalLMOutputWithPast` where every field is `None` under normal inference
conditions: there is no LM head so `logits=None`; `hidden_states=None` unless
`output_hidden_states=True` is explicitly requested; `past_key_values=None`
when `use_cache` is off; `attentions=None` by default.

`transformers.ModelOutput.__post_init__` only inserts non-`None` values into
the underlying `OrderedDict`. With all fields `None`, the `OrderedDict` is
empty. When the comparison evaluator calls `torch.utils._pytree.tree_flatten`
on this empty dict it receives zero leaves. `_compare_atol` then filters the
leaf list and calls `max([])`, raising `ValueError: max() iterable argument
is empty`.

This is purely a loader-layer bug: the loader allowed the model to be called
in a way that produces a semantically empty output, hiding the actual
computation from the test framework.

## Fix
Added `_EmbeddingWrapper(torch.nn.Module)` in
`nemotron/nemotron_embed_vl/pytorch/loader.py`. The wrapper injects
`output_hidden_states=True` into every forward call and returns
`output.hidden_states[-1]` — the final transformer hidden state — as a plain
tensor. This is the output that the model's own `encode_queries`/`encode_docs`
methods pool to produce embeddings.

Also added a test-config entry in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
(`status: EXPECTED_PASSING`) so the model is tracked by CI.

Changes:
- `tt_forge_models/nemotron/nemotron_embed_vl/pytorch/loader.py` — added `_EmbeddingWrapper`, wrapped model in `load_model`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `EXPECTED_PASSING` entry

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    125.06s (0:02:05)
- Tier A attempts: N/A

## Files changed
- tt_forge_models/nemotron/nemotron_embed_vl/pytorch/loader.py
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9af4f4148e407d86707889d1f8777b023b891a57 |
| tt-forge-models | 0da5e727e4c595ee7a6e936273c0068ac5da78bc |
