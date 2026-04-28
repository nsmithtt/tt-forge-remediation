# Remediation Summary: functionary_medium_v2_2_i1_gguf-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[functionary_medium_v2_2_i1_gguf/causal_lm/pytorch-v2.2_i1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — Mixtral-8x7B (~46.7B params, ~93 GB BF16) exceeds n150 single-device DRAM (~12 GB); three loader bugs fixed before hardware ceiling reached

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-mixtral-architecture-misidentification

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.1434917568403199. Required: pcc=0.95.

## Root cause
Three compounding loader bugs prevented correct loading and compilation of this Mixtral-8x7B GGUF.

**Bug 1 — narrow `_patched_load_gguf_checkpoint` signature (pre-existing, 26 loaders affected)**

Twenty-six GGUF loaders monkey-patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
at import time with a narrow `(gguf_path, return_tensors=False)` signature. Transformers 5.2.0
added a `model_to_load` kwarg, causing `TypeError` when any of these loaders was imported in the
same pytest session.

**Bug 2 — architecture misidentification (original PCC=0.143 failure)**

The GGUF file stores `general.architecture = "llama"` (old llama.cpp convention for Mixtral models).
`AutoModelForCausalLM.from_pretrained` treats this as standard LlamaForCausalLM, silently ignoring
the `llama.expert_count=8` metadata. All expert MLP weights (`ffn_gate_exps`, `ffn_up_exps`,
`ffn_down_exps`) are unmapped and randomly initialized. Every output token is near-random, giving
PCC ≈ 0.143.

The fix reads GGUF metadata directly via `GGUFReader` to construct a correct `MixtralConfig` and
manually maps all GGUF expert tensors (stacked `_exps` format) into the transformers 5.x
`MixtralForCausalLM` state dict layout (`mlp.experts.gate_up_proj` + `mlp.experts.down_proj`).

**Bug 3 — XLA-incompatible expert dispatch (exposed after Bug 2 fix)**

`MixtralExperts.forward` (the default "eager" implementation) uses `nonzero()` + a Python for-loop
over dynamically-discovered active experts. Running this FX graph through XLA's
`partition_fx_graph_for_cpu_fallback` (in `dynamo_bridge.py:762`) crashes the Python interpreter
with a fatal error in `torch._ops.__call__`, producing a Python process dump with no pytest summary.

Setting `model.config._experts_implementation = "batched_mm"` routes through
`batched_mm_experts_forward` from `transformers.integrations.moe`, which uses only static-shape
`torch.bmm` operations, eliminating the XLA-incompatible control flow.

**Hardware ceiling (reason for XFAIL)**

Even with all three bugs fixed, Mixtral-8x7B contains ~46.7B parameters. At bfloat16 that is ~93 GB,
far exceeding the n150's ~12 GB single-device DRAM. This follows the same pattern as the
`dolphin_27_mixtral_8x7b_gguf` and `chinese_mixtral` reports.

## Fix
All three bugs were fixed in the loader (`tt_forge_models/functionary_medium_v2_2_i1_gguf/causal_lm/pytorch/loader.py`)
on branch `remediation/functionary_medium_v2_2_i1_gguf-pytorch-Q4_K_M-single_device-inference`.

**Fix for Bug 1** (commit `b7b69ee9db`):
Changed all 26 GGUF loaders from narrow `(gguf_path, return_tensors=False)` signature to
`(*args, **kwargs)` passthrough pattern via sed, applied across the tt_forge_models repo.

**Fix for Bug 2** (commit `15c35d625a`):
Complete loader rewrite that:
- Reads GGUF metadata directly via `GGUFReader` to detect `llama.expert_count > 0`
- Constructs `MixtralConfig` with `num_local_experts=8`, `num_experts_per_tok=2`, correct dims
- Creates `MixtralForCausalLM(config)` directly (bypassing `AutoModelForCausalLM`)
- Manually dequantizes and maps all GGUF tensors including stacked expert tensors
  (`ffn_gate_exps` + `ffn_up_exps` → `mlp.experts.gate_up_proj` via `torch.cat(..., dim=1)`,
   `ffn_down_exps` → `mlp.experts.down_proj`)

**Fix for Bug 3** (commit `bdd31e2e7e`):
Added `model.config._experts_implementation = "batched_mm"` after model construction. Since
`MixtralExperts.__init__` stores `self.config = config` (a reference, not a copy), this change
propagates to all 32 × 1 expert modules at runtime. The `@use_experts_implementation` decorator
wrapping `MixtralExperts.forward` dispatches to `batched_mm_experts_forward` which uses only
static-shape `torch.bmm` operations compatible with XLA.

The test config entry `functionary_medium_v2_2_i1_gguf/causal_lm/pytorch-v2.2_i1_Q4_K_M_GGUF-single_device-inference`
was added to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` with
`status: KNOWN_FAILURE_XFAIL` (commit `003f5c65e` in tt-xla remediation branch).

## Verification
- pytest exit: not run (XFAIL disposition; silicon run not attempted after hardware ceiling identified)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/functionary_medium_v2_2_i1_gguf/causal_lm/pytorch/loader.py` (complete rewrite + batched_mm fix)
- 25 other GGUF loaders (narrow signature fix)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 003f5c65e |
| tt-forge-models | bdd31e2e7e |
