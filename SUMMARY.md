# Remediation Summary: deepseek_r1_nvfp4-causal_lm-pytorch-R1_NVFP4_V2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_nvfp4/causal_lm/pytorch-R1_NVFP4_V2-single_device-inference]

## Result
FAIL — Embedding kernel circular buffers (8 500 224 B) exceed L1 limit (1 572 864 B); Tier B, fix unknown without deeper tt-metal kernel analysis

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
embedding-cb-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(Root cause: pytest.ini missing `pythonpath = tests`, causing conftest.py to abort with
`ModuleNotFoundError: No module named 'infra'`; the SWIG warning was the last line printed.)

## Root cause

**Layer 1 — loader/tt-xla (fixed):** `pytest.ini` was missing `pythonpath = tests`.
When pytest is invoked without the custom `venv/activate` script (which adds `tests/` to
`PYTHONPATH` via its `export PYTHONPATH=...` line), `conftest.py` cannot import the local
`infra` package, and pytest aborts. The SWIG `swigvarlink` DeprecationWarning appears as the
final warning summary line after pytest exits, matching the reported failure string.

**Layer 2 — loader (fixed):** After the pytest.ini fix the test reaches the model forward
pass. The transformers 5.x `grouped_mm` MoE path calls
`torch.histc(expert_ids_g.int(), ...)` to compute expert token offsets, then uses
`torch._grouped_mm` for the expert matmul. Both operations are CUDA-specific: `torch.histc`
on `Int` dtype falls back to a CPU implementation that raises
`NotImplementedError: "histogram_cpu" not implemented for 'Int'` under XLA tracing.

**Layer 3 — tt-metal (unfixed, Tier B):** After forcing `experts_implementation="batched_mm"`
to avoid the CUDA-only MoE path, compilation reaches the embedding op. At execution time,
`tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region` throws:

```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 8500224 B which is beyond max L1 size of 1572864 B
```

The embedding layer has vocab_size = 129 280 (not reduced by the loader's dimension
overrides) and hidden_size = 1 024. The tt-metal embedding kernel allocates circular buffers
that grow to ≈ 8.5 MB per-core region, exceeding the 1.572 MB L1 limit. The CB allocation
formula and the appropriate tiling strategy for this vocab/hidden ratio are not apparent
from the surface error; deeper analysis of the embedding kernel is required.

## Fix

**tt-xla — `pytest.ini`** (committed, pushed):
Added `pythonpath = tests` so conftest.py can import `infra` regardless of how the venv is
activated. Also added `filterwarnings` entries to suppress the noisy SWIG
`swigvarlink`/`SwigPy` DeprecationWarning.

**tt-forge-models — `deepseek_r1_nvfp4/causal_lm/pytorch/loader.py`** (committed):
Added `"experts_implementation": "batched_mm"` to `model_kwargs` in `load_model`.
The `batched_mm` path uses `torch.bmm` which is XLA-compatible, avoiding the CUDA-only
`histc(Int)` + `torch._grouped_mm` calls in the `grouped_mm` path.

**Proposed fix for the embedding CB overflow (not implemented):** Investigate the tt-metal
embedding kernel's CB allocation in
`ttnn::prim::EmbeddingsDeviceOperation` / its associated program factory. The CB size needs
to be bounded by L1 regardless of vocab_size, likely by tiling the vocab dimension or
reducing the number of CB pages allocated per core. This may require changes to both the
program factory and the kernel code in tt-metal.

## Tier B justification

- **Indicator:** `internal-error-unknown-mechanism` — the surface error is `INTERNAL: Error
  code: 13` (Bad StatusOr access) and the mechanism is CB overflow, but the root cause of
  *why* the embedding kernel allocates 8.5 MB of CBs for this (vocab=129 280, hidden=1 024)
  config is unknown without instrumenting the embedding kernel's CB allocation logic in
  tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    205.99s (second run, after loader fix)
- Tier A attempts: N/A

## Files changed
- `tt-xla/pytest.ini` — added `pythonpath = tests` and SWIG filterwarnings
- `tt-xla/third_party/tt_forge_models/deepseek_r1_nvfp4/causal_lm/pytorch/loader.py` — forced `experts_implementation="batched_mm"`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1fa9ebc894e0aae9db01fe82f4ff10f93243f0f7 |
| tt-forge-models | 9945f7e24c33102d0cadc1027a50fd46fbc7beb7 |
