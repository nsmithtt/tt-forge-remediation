# Remediation Summary: huihui_qwen3_5_4b_abliterated_gguf-causal_lm-pytorch-4B_Abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_5_4b_abliterated_gguf/causal_lm/pytorch-4B_Abliterated_GGUF-single_device-inference]

## Result
FAIL — SSM scan loop compilation hang in GatedDeltaNet (Tier B: new infrastructure needed)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ssm-scan-loop-compilation-hang

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

On the `aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-0` branch the loader fix from
the previous remediation was not yet merged. 26 GGUF loaders still used the old
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` signature which
does not accept transformers 5.x's `model_to_load` kwarg. After fixing the loaders, the
model loads as `Qwen3_5ForCausalLM` and compilation begins but hangs indefinitely in the
TT-MLIR compiler (module_builder.cc) while processing the GatedDeltaNet SSM scan loop.

## Root cause

**Loader bug (fixed):** 26 GGUF loaders (on the `aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-0`
branch) still had `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` without
`**kwargs`. When `from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`,
the broken patcher rejects the argument.

The `huihui_qwen3_5_4b_abliterated_gguf` loader also lacked qwen35 architecture support:
the GGUF file uses architecture `qwen35` (a Mamba-attention SSM hybrid), which requires
`Qwen3_5ForCausalLM`, not the default `Qwen3ForCausalLM`. A context-manager load wrapper is
needed to remap `qwen35→qwen3_5_text` and synthesise `layer_types` from `full_attention_interval`.

**Compiler-stack bug (Tier B):** After all loader fixes, `Qwen3_5ForCausalLM` loads
successfully (confirmed by the `LOAD REPORT` printout). Compilation starts in
`module_builder.cc` but hangs for >9 minutes before being killed by the 900-second timeout.
The GatedDeltaNet SSM layers contain Python scan loops (`torch_chunk_gated_delta_rule`) that
TorchDynamo unrolls into a massive StableHLO scatter graph. The TT-MLIR compiler hangs
on this graph. This is the same `ssm-scan-loop-compilation-hang` bug documented in the
prior report for this test.

## Fix

**Loader fixes (tt_forge_models, remediation branch `remediation/huihui_qwen3_5_4b_abliterated_gguf-causal_lm-pytorch-4B_Abliterated_GGUF-single_device-inference-v3`):**

`f0516077b9` — Two combined changes:
1. Fixed `_patched_load_gguf_checkpoint` signature in 26 GGUF loaders: changed
   `(gguf_path, return_tensors=False)` → `(gguf_path, return_tensors=False, **kwargs)` and
   forwarded `**kwargs` to the original call.
2. Rewrote `huihui_qwen3_5_4b_abliterated_gguf/causal_lm/pytorch/loader.py`:
   - `_register_qwen35_gguf_tables()`: registers `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES`,
     `GGUF_TO_TRANSFORMERS_MAPPING`, `TENSOR_PROCESSORS` (custom `Qwen35TensorProcessor`
     for `ssm_conv1d.weight` reshape), and `GGUF_TO_FAST_CONVERTERS`.
   - `_find_real_load_gguf_checkpoint()`: BFS through the patcher chain to find the real
     transformers function (identified by `fn.__globals__ is vars(_gguf_utils)`).
   - `_qwen35_load_ctx()`: context manager that temporarily installs a corrected
     `load_gguf_checkpoint` (remaps `qwen35→qwen3_5_text`, generates `layer_types`
     from `full_attention_interval`) and patches `get_gguf_hf_weights_map`.
   - Added `use_cache=False` to avoid `Qwen3_5DynamicCache` in model output.
   - `load_shard_spec`: guarded `hasattr(layer, 'self_attn')` for hybrid layer types.
   - `requirements.txt` with `gguf>=0.10.0`.

**Proposed compiler-stack fix (not attempted — Tier B):** Implement a scan primitive
in `tt-mlir` that can compile GatedDeltaNet's recurrent scan without unrolling it into
a scatter graph. This would require new infrastructure in the StableHLO→TTIR lowering
pass.

## Tier B justification
Indicator: **new-infrastructure**. The GatedDeltaNet scan loop requires a scan/recurrent
primitive in the TT-MLIR compiler. Currently the Python loop unrolls into a massive
StableHLO scatter graph that causes the compiler to hang (>9 minutes observed). Implementing
this requires adding a new scan op and lowering path across multiple files in tt-mlir,
which is beyond the scope of a single scoped fix.

## Verification
- pytest exit: TIMEOUT (compilation hang, test killed at 900s)
- Hardware:    blackhole-p150b
- Duration:    >900s (loader fix: ~5min model load; compiler hang: >9min observed before kill)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/huihui_qwen3_5_4b_abliterated_gguf/causal_lm/pytorch/loader.py` (rewritten with SSM hybrid support)
- `tt_forge_models/huihui_qwen3_5_4b_abliterated_gguf/causal_lm/pytorch/requirements.txt` (added)
- 26 other GGUF loaders: `_patched_load_gguf_checkpoint` signature fixed to `(gguf_path, return_tensors=False, **kwargs)`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a911cc28f1aaaa2b99cb9b7e053c16828343a394 |
| tt-forge-models | f0516077b9f48ed044e7c5a6d2a67e1eae92f70 |
