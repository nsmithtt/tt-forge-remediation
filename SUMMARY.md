# Remediation Summary: got_r1_14b_i1_gguf-causal_lm-pytorch-GoT_R1_14B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[got_r1_14b_i1_gguf/causal_lm/pytorch-GoT_R1_14B_i1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Loader-layer bug. During pytest collection, all loader modules are imported
alphabetically. Several loaders (e.g. bartowski_coniccat_qwen3_5_27b_writer_gguf,
daniloreddy_qwen3_5_0_8b_gguf, dmind_3_mini_i1_gguf) install narrow-signature
monkey-patches of transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint
with signature (gguf_path, return_tensors=False). These are imported alphabetically
before got_r1_14b_i1_gguf. When the got_r1_14b_i1_gguf loader calls
AutoModelForCausalLM.from_pretrained(), transformers 5.2.0 internally calls
load_gguf_checkpoint(path, return_tensors=False, model_to_load=model), hitting
the narrow-sig patch already installed by an earlier loader. This raises TypeError
because the patched function does not accept the model_to_load keyword argument.
The got_r1_14b_i1_gguf loader itself did not install any patch, making it a
victim of cross-loader clobbering.

## Fix
In got_r1_14b_i1_gguf/causal_lm/pytorch/loader.py, added module-level code
that:
1. Temporarily pops transformers.modeling_gguf_pytorch_utils from sys.modules
   to obtain a fresh (unpatched) import of the real transformers function.
2. Immediately restores the (possibly monkey-patched) module cache entry so
   that no other code in the process is disrupted.
3. Installs a (*args, **kwargs) pass-through wrapper as load_gguf_checkpoint
   across all four module aliases that transformers uses
   (modeling_gguf_pytorch_utils, configuration_utils,
   models.auto.tokenization_auto, tokenization_utils_tokenizers).

File changed: tt-xla/third_party/tt_forge_models/got_r1_14b_i1_gguf/causal_lm/pytorch/loader.py

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    573.72s (0:09:33)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/got_r1_14b_i1_gguf/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d5ae72c9f2aabc6f3c705d2e8602d07ca82180bb |
| tt-forge-models | d8858458bf0e83cffe529e8195d8386bbe6ba456 |
