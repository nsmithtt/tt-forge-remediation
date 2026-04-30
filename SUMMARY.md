# Remediation Summary: hyperclassifier_gpt_oss_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hyperclassifier_gpt_oss_gguf/causal_lm/pytorch-Q4_K_M-single_device-inference]

## Result
XFAIL — model is 20.9B params (~41.8 GB BF16) which exceeds single-device DRAM capacity (32 GB on p150b Blackhole); loader bugs were fixed independently

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gpt-oss-gguf-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: 'Qwen3MoeExperts' object is not iterable

## Root cause
Two loader bugs were present:

1. **GGUF arch not registered (loader)**: The GGUF file has `general.architecture = "gpt-oss"` but transformers 5.2.0 did not have this architecture in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING`. The loader's `_patch_gpt_oss_support()` function existed but was not registering a complete mapping (missing `expert_feed_forward_length → moe_intermediate_size` and `attention.sliding_window → sliding_window` keys). Without the patched `load_gguf_checkpoint` being applied to all 4 call sites, the tokenizer load could succeed via stale pycache while the model load would fail.

2. **`load_shard_spec` iterates `Qwen3MoeExperts` as a `ModuleList` (loader)**: In transformers 5.2+, `Qwen3MoeSparseMoeBlock.experts` is always a `Qwen3MoeExperts` object storing 3D weight tensors (`gate_up_proj: [num_experts, 2*moe_intermediate_dim, hidden_dim]`, `down_proj: [num_experts, hidden_dim, moe_intermediate_dim]`). The `load_shard_spec` method iterated over `experts` assuming it was a `nn.ModuleList`, causing the `TypeError`. The fix detects `hasattr(experts, 'gate_up_proj')` and shards the 3D tensors directly.

After both loader fixes, the silicon run failed with `INTERNAL: Error code: 13` — this is the TT device OOM error. The model is 20.9B parameters which requires approximately 41.8 GB in BF16. The p150b Blackhole device has 32 GB DRAM, so the model exceeds the single-device capacity ceiling.

## Fix
**tt-forge-models** (`third_party/tt_forge_models/hyperclassifier_gpt_oss_gguf/causal_lm/pytorch/loader.py`):
- Added complete GGUF key mappings for `expert_feed_forward_length → moe_intermediate_size` and `attention.sliding_window → sliding_window` in `_patch_gpt_oss_support()`
- Wrapped `load_gguf_checkpoint` with `_patched_load_gguf_checkpoint` and applied the patch to all 4 call sites (`_gguf_utils`, `_config_utils`, `_auto_tokenizer`, `_tok_utils`)
- Fixed `load_shard_spec` to detect `Qwen3MoeExperts` (via `hasattr(experts, 'gate_up_proj')`) and shard 3D weight tensors directly rather than iterating as a `ModuleList`

**tt-xla** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `KNOWN_FAILURE_XFAIL` entries for both `Q4_K_M` and `i1_Q4_K_M` variants with hardware capacity reason

## Verification
- pytest exit: FAIL (INTERNAL: Error code: 13 — device OOM)
- Hardware:    blackhole-p150b
- Duration:    1315.88s (0:21:55)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hyperclassifier_gpt_oss_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 24db9d19355fe0b8eb2599da0ba6aa0e77069fdc |
| tt-forge-models | 8a885905576f373be20a78fa169e8a131f1e3673 |
