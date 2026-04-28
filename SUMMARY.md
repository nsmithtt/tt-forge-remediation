# Remediation Summary: abcorrea_bw_v1-causal_lm-pytorch-bw_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[abcorrea_bw_v1/causal_lm/pytorch-bw_v1-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
peft-modules-to-save-wrapper-tie-weights-conflict

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
tests/infra/runners/torch_device_runner.py:134: in _safely_put_workload_on_device
    workload.model.tie_weights()
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:2542: in tie_weights
    setattr(parent, name, source_param)
venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1989: in __setattr__
    self.register_parameter(name, value)
venv/lib/python3.12/site-packages/torch/nn/modules/module.py:616: in register_parameter
    raise KeyError(f"attribute 'weight' already exists")
KeyError: "attribute 'weight' already exists"
```

## Root cause
`abcorrea/bw-v1` is a LoRA fine-tune of `Qwen/Qwen3-4B-Thinking-2507` with
`modules_to_save=['lm_head', 'embed_tokens']` in its PEFT adapter config. When
`AutoModelForCausalLM.from_pretrained("abcorrea/bw-v1")` is called, recent
transformers (4.50+) auto-detects `adapter_config.json` and loads the model via
`PeftAdapterMixin`, wrapping `lm_head` and `embed_tokens` in PEFT's
`ModulesToSaveWrapper`. The test framework then calls `workload.model.tie_weights()`
(required by torch-xla) which invokes `setattr(lm_head, 'weight', embed_tokens.weight)`.
`nn.Module.__setattr__` calls `register_parameter('weight', ...)`, which raises
`KeyError` because `ModulesToSaveWrapper.__getattr__` delegates to the active
sub-module, making `hasattr(lm_head, 'weight')` True via proxy — while `'weight'`
is not in the wrapper's own `_parameters` dict, causing the check
`hasattr(self, name) and name not in self._parameters` to fire.

## Fix
Changed `abcorrea_bw_v1/causal_lm/pytorch/loader.py` in `tt-forge-models` to
load the model via `peft.AutoPeftModelForCausalLM.from_pretrained()` followed
immediately by `.merge_and_unload()`. This collapses the LoRA adapter matrices
and unwraps `ModulesToSaveWrapper` for `modules_to_save` modules, returning a
plain `Qwen3ForCausalLM` with fine-tuned weights baked in. After the merge,
`lm_head` is a regular `nn.Linear` and `tie_weights()` succeeds.

File changed: `abcorrea_bw_v1/causal_lm/pytorch/loader.py`
Repo: `tt-forge-models`
Branch: `remediation/abcorrea_bw_v1-causal_lm-pytorch-bw_v1-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    149.27s
- Tier A attempts: N/A

## Files changed
- `abcorrea_bw_v1/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7fe52b4b18c29bbeb69f49162e32371e5e1d3e84 |
| tt-forge-models | 85f5b86ae73607ea585df79dbcaa80cd840cc930 |
