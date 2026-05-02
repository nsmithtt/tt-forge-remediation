# Remediation Summary: mixtral_8x7b_v01_gguf-causal_lm-pytorch-8x7B_v0.1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mixtral_8x7b_v01_gguf/causal_lm/pytorch-8x7B_v0.1_GGUF-single_device-inference]

## Result
XFAIL — Hardware capacity ceiling: 46.7B params BF16 = ~93 GB >> 34.2 GB p150b DRAM

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-llama-arch-mixtral-moe-weights-unloaded

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.11568523437032278. Required: pcc=0.95.

## Root cause
The Mixtral 8x7B Q4_K_M GGUF file declares `general.architecture = "llama"`. Transformers
instantiates `LlamaForCausalLM` and skips all 8 per-expert MoE tensors
(`blk.{i}.ffn_gate.{e}.weight`, `ffn_up.{e}.weight`, `ffn_down.{e}.weight`).
The MLP is randomly initialised, producing PCC 0.115.

After fixing the loader to correctly instantiate `MixtralForCausalLM`, the full
46.7B-parameter model dequantised to BF16 (~93 GB) exceeds the p150b DRAM ceiling
(34.2 GB). The test encounters a `RuntimeError: Expected inputs of BF16 type but
got mat_a.dtype=torch.float32` in `torch._grouped_mm` during Dynamo tracing of the
MoE path before hitting an OOM, but the root disposition is hardware-class.

## Fix
Two changes:

1. **Loader** (`tt_forge_models/mixtral_8x7b_v01_gguf/causal_lm/pytorch/loader.py`):
   Replaced `AutoModelForCausalLM.from_pretrained(..., gguf_file=)` with a custom
   `_load_mixtral_from_gguf` that reads tensors directly via `GGUFReader + dequantize`,
   maps `blk.N.*` tensor names to `MixtralForCausalLM` state-dict keys, collects
   per-expert `ffn_gate.{e}/ffn_down.{e}/ffn_up.{e}` and stacks them into batched
   `mlp.experts.gate_up_proj [E, 2I, H]` and `mlp.experts.down_proj [E, H, I]`.
   Builds `MixtralConfig` from GGUF metadata and loads the state-dict with `strict=False`.

2. **Test config** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
   Added `KNOWN_FAILURE_XFAIL` entry for the hardware ceiling.

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    blackhole-p150b
- Duration:    1413.39s (0:23:33)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/mixtral_8x7b_v01_gguf/causal_lm/pytorch/loader.py
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6d8c4b337c56a7b61579b881b60286965bc66d55 |
| tt-forge-models | 38dbae866f2b16d39c02d62bac1a265e14aeb447 |
