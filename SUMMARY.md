# Remediation Summary: lfm2_5_gguf/causal_lm/pytorch-1_2B_Instruct_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lfm2_5_gguf/causal_lm/pytorch-1_2B_Instruct_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
lfm2-gguf-hybrid-conv-cache-use-cache

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache

(Proximate failure before this was KeyError: 'lfm2' in GGUF_TO_FAST_CONVERTERS,
and TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument
'model_to_load' from stale patches installed by other loaders in the same session.)

## Root cause
Three loader bugs in lfm2_5_gguf/causal_lm/pytorch/loader.py:

1. KeyError: 'lfm2' -- GGUF_TO_FAST_CONVERTERS in transformers lacked an
   entry for the 'lfm2' architecture. LFM2.5 uses a GPT2-style BPE tokenizer
   (tokenizer.ggml.model = "gpt2") but the converter was not registered.
   Fixed in hf-bringup-6 commit 4c11015d80.

2. TypeError: _patched_load_gguf_checkpoint() got unexpected kwarg 'model_to_load'
   -- other GGUF loaders in the same pytest session had installed a narrow-signature
   wrapper for load_gguf_checkpoint. When transformers 5.2.0 added model_to_load
   as a kwarg, the old wrapper broke. Fixed in hf-bringup-6 commit 4a33e09649
   by updating all 26 wrappers to (*args, **kwargs).

3. TypeError: equal() argument must be Tensor, not Lfm2HybridConvCache --
   Lfm2HybridConvCache is not a registered pytree node. The XLA comparison
   evaluator calls torch.equal() on every output leaf of the model, which fails
   when the KV cache object is included in the model output. Setting use_cache=False
   prevents the cache from appearing in the model output.

## Fix
The remediation is built on top of hf-bringup-6 (which already fixed bugs 1 and 2
for the lfm2_5_gguf loader via commit 4c11015d80 and the global
_patched_load_gguf_checkpoint fix in 4a33e09649).

Single additional commit in tt-forge-models:
- Branch: remediation/lfm2_5_gguf-causal_lm-pytorch-1_2B_Instruct_GGUF-single_device-inference-v2
- Commit 66b67f5a09: Add use_cache=False to load_inputs() in
  lfm2_5_gguf/causal_lm/pytorch/loader.py

The tt_forge_models pointer in tt-xla was updated via:
- Branch: remediation/lfm2_5_gguf-causal_lm-pytorch-1_2B_Instruct_GGUF-single_device-inference
- Commit fd6fee0ed: Update third_party/tt_forge_models to 66b67f5a09

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    171.19s (0:02:51)
- Tier A attempts: N/A

## Files changed
- tt-forge-models/lfm2_5_gguf/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fd6fee0ed63991e6c32bfec516d9624ff68f0a3e |
| tt-forge-models | 66b67f5a099f05dd34d574a3ac49c8aecc77c032 |
