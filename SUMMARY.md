# Remediation Summary: csai-gpt-oss-20b-gguf-causal-lm-pytorch-20b-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[csai_gpt_oss_20b_gguf/causal_lm/pytorch-20B_GGUF-single_device-inference]

## Result
XFAIL — 20B Q4_K_M GGUF (~11.25 GB) exhausts N150 single-device DRAM (12 GB); OOM allocating ~1 GB activation buffer at runtime

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-oom-20b-q4km-n150-dram-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_FATAL: Out of Memory: Not enough space to allocate 1061683200 B DRAM buffer across 8 banks,
where each bank needs to store 132710400 B, but bank size is 4273390016 B
(allocated: 4221955392 B, free: 51434624 B, largest free block: 45589184 B)
```
Raised during `tt::runtime::ttnn::submit` when attempting to allocate an activation buffer
after model weights have consumed nearly all of the N150's 12 GB DRAM.

The originally reported `TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote
mmio device` messages are non-fatal: they are caught and logged as warnings in current
tt-metal. They were a red herring; the real failure is the DRAM OOM above.

## Root cause
The csai-gpt-oss-20b GGUF is a 20B-parameter Qwen3-MoE model quantized to Q4_K_M,
producing a checkpoint of approximately 11.25 GB. Loading onto a single N150 (12 GB DRAM)
leaves roughly 49 MB free per bank — far below the ~1 GB needed for the first large
activation buffer allocated during graph execution. This is a hardware capacity ceiling,
not a compiler bug.

Three loader bugs were also found and fixed along the way (see Fix section), but none of
them was the root cause of the OOM.

## Fix
Three loader-layer fixes were committed to tt_forge_models on the remediation branch,
and the test was marked KNOWN_FAILURE_XFAIL in tt-xla:

**tt_forge_models — commit 2559d167f2**
Cherry-picked fix for `_patched_load_gguf_checkpoint` across 26 GGUF loaders that
monkey-patched the function at module-level with signature `(gguf_path, return_tensors=False)`.
Transformers 5.2.0 now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`,
causing a `TypeError`. Fix adds `model_to_load=None` and forwards it.
- Files: `csai_gpt_oss_20b_gguf/causal_lm/pytorch/loader.py` and 25 other GGUF loaders

**tt_forge_models — commit a875e38ffe**
`load_shard_spec` assumed dense MLP structure (`up_proj`, `gate_proj`, `down_proj`) but
csai-gpt-oss-20b uses Qwen3MoeExperts with batched parameters (`gate_up_proj`,
`down_proj` as `nn.Parameter` tensors). Fix uses `hasattr(mlp, "experts")` check and
accesses the batched parameters directly.
- File: `csai_gpt_oss_20b_gguf/causal_lm/pytorch/loader.py`

**tt_forge_models — commit 7d25d4868d**
With `_experts_implementation=None` (default), Qwen3MoE uses a Python for-loop over
experts that XLA cannot statically trace, causing a segfault in
`partition_fx_graph_for_cpu_fallback`. Fix sets `model.config._experts_implementation =
"batched_mm"` after `.eval()`.
- File: `csai_gpt_oss_20b_gguf/causal_lm/pytorch/loader.py`

**tt-xla — commit cded14221**
Added KNOWN_FAILURE_XFAIL to the single-device inference test config.
- File: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: FAIL (INTERNAL error 13 / DRAM OOM)
- Hardware:    n150
- Duration:    ~20 min (model loading + graph compilation before OOM)
- Tier A attempts: N/A

## Files changed
- `csai_gpt_oss_20b_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)
- 25 other GGUF loader files for model_to_load fix (tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit      |
|-----------------|-------------|
| tt-metal        | 3fa4d75355  |
| tt-mlir         | 553c0632b   |
| tt-xla          | cded14221   |
| tt-forge-models | 7d25d4868d  |
