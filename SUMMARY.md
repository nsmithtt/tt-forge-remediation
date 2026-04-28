# Remediation Summary: bartowski_goppa_ai_goppa_logillama_gguf-causal_lm-pytorch-GOPPA_LOGILLAMA_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_goppa_ai_goppa_logillama_gguf/causal_lm/pytorch-GOPPA_LOGILLAMA_Q4_K_M_GGUF-single_device-inference]

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

(CI symptom: what():  Timeout waiting for ARC msg request queue. — Blackhole device was in a stale state from a preceding hung test; after device reset the model runs correctly.)

## Root cause
When pytest collects multiple GGUF loader modules, several of them install a module-level monkey-patch on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with the fixed signature `(gguf_path, return_tensors=False)`. Transformers 5.x added a `model_to_load=None` keyword argument to this function. When the goppa_logillama test runs in a full session, a previously-collected loader's patcher is active. `AutoModelForCausalLM.from_pretrained` calls `load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)`, which hits the patcher and raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

The CI run's "ARC msg request queue" timeout indicates the CI Blackhole device was in a hung state (from a prior test), not a bug in this model. After device reset, the test passes.

## Fix
Cherry-picked commit `57caeafc70` from `origin/remediation/anubis-mini-gguf-fix-kwargs-compat` in tt-forge-models onto the remediation branch. This fix updates all 26 affected GGUF loader files to:
1. Change signature from `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):`
2. Forward `**kwargs` to `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`

Files: 26 GGUF loader files in `tt-forge-models/` (bartowski_goppa_ai_goppa_logillama_gguf is not one of the patching loaders — it uses `AutoModelForCausalLM.from_pretrained` directly, but benefits from the fix as a downstream consumer).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    273.45s (0:04:33)
- Tier A attempts: N/A

## Files changed
- 26 GGUF loader files in `tenstorrent/tt-forge-models` (via cherry-pick of `57caeafc70`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 517e1bc1e54596ee811f69c5acf43509d5267743 |
| tt-forge-models | a48de7f67d687cb0433637fbac33cf42e33ed507 |
