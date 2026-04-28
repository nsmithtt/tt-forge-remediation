# Remediation Summary: agentic-qwen-30b-a3b-i1-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[agentic_qwen_30b_a3b_i1_gguf/causal_lm/pytorch-30B_A3B_i1_GGUF-single_device-inference]

## Result
XFAIL — 30B model dequantized to BF16 (~60 GB) far exceeds single n150 DRAM capacity

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-30b-bf16-exceeds-n150-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two failures were found and addressed:

**Failure 1 (loader bug, fixed):**
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

**Failure 2 (hardware-class ceiling, XFAIL):**
```
Fatal Python error: Segmentation fault
```
Occurring in `partition_fx_graph_for_cpu_fallback` →
`UnsupportedNodesCollector.run()` → op dispatch on TT device during
the first execution attempt to identify CPU fallback ops.

## Root cause

**Loader bug:** 26 GGUF loaders (qwen35-family) monkey-patched
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import
time with a narrow signature `(gguf_path, return_tensors=False)`.
Transformers 5.2.0 now passes `model_to_load=dummy_model` as a keyword
argument, causing a `TypeError` in any test in the pytest session where
one of these loaders was collected before the agentic_qwen test ran.

**Hardware-class ceiling:** The Agentic-Qwen-30B-A3B-i1 GGUF model
(18.5 GB on disk, Q4_K_M) is loaded and dequantized to BF16 by
`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`.
The resulting model has ~60 GB of parameters in BF16 (30B × 2 bytes)
plus framework overhead, totalling ~120 GB RSS on CPU. When the
TorchXLA dynamo bridge (`extract_compiled_graph` in dynamo_bridge.py)
calls `partition_fx_graph_for_cpu_fallback`, it runs
`UnsupportedNodesCollector(*xla_args)` which executes the model forward
pass on the TT device to discover which ops need CPU fallback. The TT
n150 device (~12 GB L2 SRAM + 8 GB DRAM) cannot accommodate ~60 GB of
BF16 weights, causing a `Fatal Python error: Segmentation fault` in the
XLA runtime.

This is not a compiler bug: the model simply exceeds single-device DRAM
by a factor of ~3× (60 GB weights vs. ~20 GB device capacity).

## Fix

**Loader fix (tt-forge-models, branch `remediation/agentic-qwen-30b-a3b-i1-gguf-single-device-inference`):**
Changed all 26 affected GGUF loaders from narrow signature:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to passthrough signature:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
Commit: `b312f3a12669c312692b83777cc8241c7b9c4926`

**XFAIL config (tt-xla, branch `remediation/agentic-qwen-30b-a3b-i1-gguf-single-device-inference`):**
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`.
Commit: `850addc0f1fc743e2ab61d35f16202a6d6bb322e`

## Verification
- pytest exit: FAIL (Segmentation fault / exit 139)
- Hardware:    n150
- Duration:    23:34 (model loading ~14 min, device init + crash ~4 min after that)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_bartleby_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_crow_4b_opus_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_luna_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_huihui_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9136b98930595fefdf2a406207dfda51ef26e5f5 |
| tt-forge-models | b312f3a12669c312692b83777cc8241c7b9c4926 |
