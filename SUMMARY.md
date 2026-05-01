# Remediation Summary: gpt_oss_20b_derestricted_gguf-causal_lm-pytorch-20B_Derestricted_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_derestricted_gguf/causal_lm/pytorch-20B_Derestricted_GGUF-single_device-inference]

## Result
XFAIL — GPT-OSS 20B BF16 weights (~38 GB) fill p150b device DRAM (34 GB); only 154 MB free at inference time, insufficient for 1 GB activation buffers; hardware capacity ceiling

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, qwen3moe-experts-dynamic-loop-xla-segfault, qwen3moe-shard-spec-moe-block, hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'

After load_shard_spec fix: Fatal Python error: Segmentation fault in partition_fx_graph_for_cpu_fallback (nonzero/for-loop in Qwen3MoeExperts.forward executing with TorchFunctionMode active)

After Qwen3MoeExperts bmm fix: RuntimeError: TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 1061683200 B DRAM buffer across 8 banks (allocated: 4111272896 B, free: 162117120 B, largest free block: 66355200 B)

## Root cause
Three bugs fixed in sequence:

1. Loader layer - GGUF global patch missing model_to_load: 26 loaders install _patched_load_gguf_checkpoint at import time with signature (gguf_path, return_tensors=False). Transformers 5.x added model_to_load=None kwarg to load_gguf_checkpoint. When any GGUF model's from_pretrained runs, it passes model_to_load=dummy_model; the broken patch in the chain receives an unexpected keyword argument -> TypeError. Fixed by updating all 26 loaders to **kwargs.

2. Loader layer - load_shard_spec accessing wrong attribute on MoE layer: The gpt_oss_20b_derestricted loader's load_shard_spec accessed layer.mlp.up_proj.weight. When loaded via the gpt_oss_swallow patcher chain, the model resolves as Qwen3MoeForCausalLM, where MoE layers have Qwen3MoeSparseMoeBlock instead of a dense MLP. Qwen3MoeSparseMoeBlock stores expert weights as 3D tensors experts.gate_up_proj [num_experts, 2*intermediate_dim, hidden_dim] and experts.down_proj [num_experts, hidden_dim, intermediate_dim] - no per-layer up_proj/gate_proj/down_proj. Fixed by guarding with hasattr(mlp, "experts") and using the experts tensors directly.

3. tt-xla compiler frontend - Qwen3MoeExperts.forward nonzero/for-loop segfault: The default Qwen3MoeExperts.forward uses a data-dependent nonzero() + Python for-loop + index_add_ pattern. When partition_fx_graph_for_cpu_fallback runs with TorchFunctionMode (TorchFunctionOverride) active in the process, the XLA FX graph executor attempts to run this code with symbolic tensors, triggering a segfault. Fixed by monkey-patching Qwen3MoeExperts.forward with a device-friendly implementation: CPU path keeps the per-expert loop (safe for golden reference), device path uses a dense bmm over all experts simultaneously (static graph, no nonzero/for-loop).

4. Hardware capacity: After the above fixes, the model compiled and began executing on TT silicon. GPT-OSS 20B dequantizes to BF16 at load time, consuming ~96% of p150b's 34 GB DRAM (4111 MB of 4273 MB per bank). An intermediate activation tensor of 1.06 GB cannot be allocated (largest free block = 63 MB) -> OOM. This is the same hardware-class ceiling that caused the non-GGUF gpt_oss/pytorch-20B to be EXCLUDE_MODEL.

## Fix
Loader fixes (tt-forge-models, commit bea8016f1c):
- Added **kwargs to _patched_load_gguf_checkpoint in 26 GGUF loaders so model_to_load propagates through the patcher chain.
- Fixed load_shard_spec in gpt_oss_20b_derestricted_gguf/causal_lm/pytorch/loader.py to detect Qwen3MoeSparseMoeBlock via hasattr(mlp, "experts") and access mlp.experts.gate_up_proj / mlp.experts.down_proj for MoE layers.
- Added gpt_oss_20b_derestricted_gguf/causal_lm/pytorch/requirements.txt with gguf>=0.10.0.

tt-xla compiler fix (commit 16c16df82):
- Added _qwen3moe_experts_forward() to tt-xla/python_package/tt_torch/torch_overrides.py: replaces nonzero/for-loop with dense bmm on device path; patches Qwen3MoeExperts.forward at module load time.

Test config XFAIL (commit 16c16df82):
- Added KNOWN_FAILURE_XFAIL entry for this test in tests/runner/test_config/torch/test_config_inference_single_device.yaml.

## Tier B justification (FAIL with Tier=B only - omit otherwise)

## Verification
- pytest exit: FAIL (OOM - hardware capacity)
- Hardware:    blackhole-p150b
- Duration:    1126.53s (0:18:46) for the final OOM run
- Tier A attempts: 3 (GGUF kwargs fix, shard_spec MoE fix, Qwen3MoeExperts bmm fix; all passed; OOM is hardware-class)

## Files changed
tt-forge-models (bea8016f1c):
- gpt_oss_20b_derestricted_gguf/causal_lm/pytorch/loader.py - MoE-aware load_shard_spec + **kwargs fix
- gpt_oss_20b_derestricted_gguf/causal_lm/pytorch/requirements.txt - new file, gguf>=0.10.0
- 26 other */causal_lm/pytorch/loader.py files - added **kwargs to _patched_load_gguf_checkpoint

tt-xla (16c16df82):
- python_package/tt_torch/torch_overrides.py - Qwen3MoeExperts dense bmm forward
- tests/runner/test_config/torch/test_config_inference_single_device.yaml - KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 16c16df8285bf50a9345c6a10416a774e231f46a |
