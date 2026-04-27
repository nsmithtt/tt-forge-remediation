# Remediation Summary: bartowski_huihui_gpt_oss_20b_bf16_abliterated_gguf

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_huihui_gpt_oss_20b_bf16_abliterated_gguf/causal_lm/pytorch-HUIHUI_GPT_OSS_20B_BF16_ABLITERATED_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — segfault in tt-xla compiler frontend during `partition_fx_graph_for_cpu_fallback`

## Failure
Fatal Python error: Segmentation fault

## Root cause
Two bugs were present, one masking the other:

**Bug 1 (loader, fixed):** All GGUF loaders that monkey-patch `load_gguf_checkpoint` to add
custom architecture support (qwen35, gpt-oss) defined `_patched_load_gguf_checkpoint` with
signature `(gguf_path, return_tensors=False)`. Transformers now calls `load_gguf_checkpoint`
with `model_to_load=dummy_model`, causing `TypeError` before the model could load.
Additionally, `bartowski_huihui_gpt_oss_20b_bf16_abliterated_gguf` itself had no gpt-oss
architecture registration even though its GGUF file uses the `gpt-oss` architecture.

**Bug 2 (compiler frontend, unfixed):** After the loader fix, the test proceeds to hardware
execution and crashes with SIGSEGV. The crash occurs inside
`tt_torch/torch_overrides.py:34 __torch_function__` while
`torch_xla/_dynamo/dynamo_bridge.py:762 partition_fx_graph_for_cpu_fallback` executes the
Qwen3 MoE (GptOss) model on CPU to partition the FX graph. The C++ op dispatch at
`torch/_ops.py:841` crashes when `func(*args)` is called for an op in the MoE expert path.

The crash is in the **compiler frontend** layer (tt-xla). The `TorchFunctionOverride` and
the `_experts_forward` / `_router_forward` / `_sparse_mlp_forward` monkey patches in
`torch_overrides.py` were written to target transformers 4.57.1 GptOss interfaces. The
current transformers applies `@use_experts_implementation(is_transposed=True, has_bias=True)`
to `GptOssExperts`, which wraps the forward method at class-definition time. The monkey
patches bypass this dispatch (intentionally), but the combination of the `__torch_function__`
override and the MoE CPU execution path triggers a C-level crash.

Also relevant: the GGUF-to-transformers weight mapping for the `gpt-oss` architecture does
not map the MoE expert tensors (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`) — they
are randomly initialized. The load report shows 5 categories of weights MISSING for all 24
layers. This may interact with the crash.

## Fix
**Loader fix applied (in tt-forge-models):**
- Added `model_to_load=None` parameter to all 26 `_patched_load_gguf_checkpoint` functions
  across GGUF loaders and threaded it through to `_orig_load_gguf_checkpoint`.
- Added gpt-oss architecture registration and `_patched_load_gguf_checkpoint` to the
  `bartowski_huihui_gpt_oss_20b_bf16_abliterated_gguf` loader (previously absent).
- Not a forbidden workaround: this is a genuine transformers API update fix.

**Compiler bug (not fixed, needs tt-xla work):**
Proposed fix: in `tt_torch/torch_overrides.py`, investigate why the monkey-patched
`_experts_forward` CPU path crashes at a C++ op. Specifically:
1. Identify which op crashes using `TORCH_SHOW_CPP_STACKTRACES=1` or a core dump.
2. The GGUF mapping for gpt-oss may need to be extended to map the expert weight tensors
   (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`) from GGUF format to
   transformers format. These are currently randomly initialized (MISSING from checkpoint).
3. The `_experts_forward` device path uses `torch.bmm` with shapes derived from the
   expert weights; if the expert weights have wrong shapes due to GGUF loading issues,
   this could produce invalid tensor shapes that crash the C++ kernel.

## Verification
pytest exited with segfault (Fatal Python error) — no pytest summary line was produced.
Crash occurred at ~17m18s after test start (23:18:42 → 23:36:00).
Hardware: n150 (single device).

## Files changed
- `bartowski_huihui_gpt_oss_20b_bf16_abliterated_gguf/causal_lm/pytorch/loader.py` — added gpt-oss architecture patch + `_patched_load_gguf_checkpoint`
- 26 other `*/causal_lm/pytorch/loader.py` files — added `model_to_load=None` param

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b59228a07d8cbc4c5db59e645e9c6dcb65919d08 |
| tt-forge-models | 45866b7a1b48f7cf5a06d81705c0d8740453473b |
