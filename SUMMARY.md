# Remediation Summary: mradermacher-agentic-qwen-30b-a3b-i1-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_agentic_qwen_30b_a3b_i1_gguf/causal_lm/pytorch-30B_A3B_I1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 30B-param Qwen3-MoE model dequantized to BF16 (~60 GB) exceeds p150b DRAM (~34 GB); two prior bugs fixed before reaching hardware limit

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-30b-model-exceeds-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Three sequential issues were found:

**Bug 1 (loader):** 26 Qwen3.5 loaders patched `load_gguf_checkpoint` with a narrow
`(gguf_path, return_tensors=False)` signature. transformers 5.2.0 added a `model_to_load=`
keyword argument to that function. During pytest collection all loaders are imported, so
any Qwen3.5 loader imported before this test runs clobbers the global
`_gguf_utils.load_gguf_checkpoint` with the narrow-sig wrapper. When
`AutoModelForCausalLM.from_pretrained` calls it with `model_to_load=`, the
`TypeError` is raised.

**Bug 2 (loader):** After bug 1 was fixed, `partition_fx_graph_for_cpu_fallback` in
dynamo's XLA bridge crashed (segfault) during `torch.compile` tracing. The default
`Qwen3MoeExperts.forward` dispatches experts via a Python for-loop over a
dynamically-sized `expert_hit` tensor, which XLA cannot statically trace.

**Terminal (hardware-class):** After both bugs were fixed, the test attempted to run
inference on the TT device. The 30B-parameter model dequantized to BF16 requires
approximately 60 GB of device DRAM, but p150b provides only ~34 GB (8 banks ×
~4.27 GB/bank). OOM was observed at 34.1 GB allocated with only ~80 MB free:
```
TT_FATAL: Out of Memory: Not enough space to allocate 805306368 B DRAM buffer
across 8 banks, where each bank needs to store 100663296 B, but bank size is
4273390016 B (allocated: 4263102016 B, free: 10288000 B)
```

## Fix
**Bug 1** — fixed in `tt_forge_models` remediation branch: changed all 26
`_patched_load_gguf_checkpoint` functions from narrow signature
`(gguf_path, return_tensors=False)` to `(*args, **kwargs)` passthrough.
Files: all 26 Qwen3.5 loaders under `tt_forge_models/`.

**Bug 2** — fixed in `tt_forge_models` remediation branch:
`mradermacher_agentic_qwen_30b_a3b_i1_gguf/causal_lm/pytorch/loader.py` —
added `model.config._experts_implementation = "batched_mm"` after
`AutoModelForCausalLM.from_pretrained(...)`. The `batched_mm_experts_forward`
implementation uses only static tensor operations and is fully XLA-compatible.

**Hardware-class XFAIL** — added to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla:
```yaml
mradermacher_agentic_qwen_30b_a3b_i1_gguf/causal_lm/pytorch-30B_A3B_I1_Q4_K_M_GGUF-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "Out of Memory: 30B-param model dequantized to BF16 (~60 GB) exceeds single-device DRAM (~34 GB on p150b)"
```

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling after loader bugs fixed)
- Hardware:    blackhole-p150b
- Duration:    1508.11s (0:25:08)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mradermacher_agentic_qwen_30b_a3b_i1_gguf/causal_lm/pytorch/loader.py` — batched_mm fix
- `tt_forge_models/<26 Qwen3.5 loaders>/causal_lm/pytorch/loader.py` — narrow-sig fix
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0486333a325a83800c786462aafa5d1261d38eea |
| tt-forge-models | 6f54943491ac6d6e34284f1df0f2841e6099c154 |
