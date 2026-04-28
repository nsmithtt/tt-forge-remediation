# Remediation Summary: dolphin_27_mixtral_8x7b_gguf-causal_lm-pytorch-2.7_Mixtral_8x7B_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dolphin_27_mixtral_8x7b_gguf/causal_lm/pytorch-2.7_Mixtral_8x7B_GGUF-single_device-inference]

## Result
XFAIL — Mixtral 8x7B GGUF (Q4_K_M ~24 GB) exceeds n150 single-device DRAM (~12 GB); two loader bugs fixed before hardware ceiling reached

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.1281011268198003. Required: pcc=0.95.

## Root cause
Two compounding loader bugs prevented correct loading of this Mixtral 8x7B GGUF file.

**Bug 1 — architecture misidentification (original failure, PCC=0.128)**

The GGUF file stores `general.architecture = "llama"` (old llama.cpp convention for Mixtral
models). Transformers 5.2.0's `load_gguf_checkpoint` maps this to `LlamaConfig`, silently
dropping the `num_local_experts` and related MoE fields. `AutoModelForCausalLM` thus instantiates
`LlamaForCausalLM` (no expert modules), the 8×8 expert weight tensors from the GGUF file are
completely unmapped and skipped, and all FFN output is zero/random → PCC ≈ 0.128.

**Bug 2 — missing tensor processor for per-expert GGUF format (exposed after Bug 1 fix)**

After passing `config=MixtralConfig(...)` to force the correct model class, `load_gguf_checkpoint`
calls `get_gguf_hf_weights_map(model_to_load, processor)` which looks up `model_type="mixtral"` in
gguf-py's `MODEL_ARCH_NAMES` dict — that dict has no "mixtral" entry, so it raises
`NotImplementedError`.

Additionally, this GGUF file uses the *old* per-expert format
(`blk.{bid}.ffn_{gate,down,up}.{eid}.weight`) rather than the newer stacked format
(`blk.{bid}.ffn_gate_exps`). Transformers has no `TensorProcessor` that maps these per-expert
GGUF tensors to `MixtralForCausalLM`'s `block_sparse_moe.experts.{j}.w{1,2,3}.weight` parameters.

**Hardware ceiling (reason for XFAIL)**

Even with both loader bugs fixed, Mixtral 8x7B contains ~46.7B parameters. The Q4_K_M GGUF
dequantizes to ~93 GB in BF16, far exceeding the n150's ~12 GB single-device DRAM.

## Fix
Both bugs were fixed in the loader (`tt_forge_models/dolphin_27_mixtral_8x7b_gguf/causal_lm/pytorch/loader.py`).

**Fix for Bug 1** (commit `31682b021b` on `remediation/dolphin-27-mixtral-8x7b-gguf-causal-lm`):
- Added `_build_mixtral_config_from_gguf()` which reads GGUF metadata via `GGUFReader` to detect
  `llama.expert_count > 0` and constructs an explicit `MixtralConfig` with correct MoE parameters.
- Passes `config=MixtralConfig(...)` to `AutoModelForCausalLM.from_pretrained`, forcing
  `MixtralForCausalLM` to be instantiated.
- Sets `model.config._experts_implementation = "batched_mm"` to avoid XLA-incompatible dynamic
  for-loop in `MixtralExperts.forward` (same pattern as Chinese Mixtral report).

**Fix for Bug 2** (commit `de66b59879` on `remediation/dolphin-27-mixtral-8x7b-gguf-causal-lm`):
- Added `_patch_transformers_mixtral_gguf()` called at import time, which:
  1. Patches `get_gguf_hf_weights_map` to remap `model_type="mixtral"` → `"llama"` before the
     gguf-py architecture lookup (mirrors existing hacks for "cohere", "qwen2_moe", etc.).
  2. Defines `MixtralTensorProcessor(LlamaTensorProcessor)` which intercepts old-format per-expert
     tensors (`blk.{bid}.ffn_{gate,down,up}.{eid}.weight`) in `process()`, assigns them directly
     to `parsed_parameters["tensors"]["model.layers.{bid}.block_sparse_moe.experts.{eid}.w{1,2,3}.weight"]`,
     and returns `GGUFTensor(weights, None, {})` to signal "already handled".
     Weight mapping: `ffn_gate.{eid}` → `w1`, `ffn_up.{eid}` → `w3`, `ffn_down.{eid}` → `w2`.
  3. Patches `load_gguf_checkpoint` to replace `TENSOR_PROCESSORS["llama"]` with
     `MixtralTensorProcessor` for the duration of any call where `model_to_load.config.model_type == "mixtral"`.

**Fix for Bug 3** (commit `b0d3597869` on `remediation/dolphin-27-mixtral-8x7b-gguf-causal-lm`):
- Fixed hardcoded `vocab_size=32000` in `_build_mixtral_config_from_gguf()`: this GGUF has
  32002 tokens (`tokenizer.ggml.tokens` array). Now reads vocab_size from that field,
  falling back to `llama.vocab_size` then 32000.

The XFAIL entry was added in tt-xla at
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
(commit `bd3f880685` on `remediation/dolphin-27-mixtral-8x7b-gguf-causal-lm`).

## Verification
- pytest exit: XFAIL
- Hardware:    n150
- Duration:    67.31s (0:01:07)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/dolphin_27_mixtral_8x7b_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f8d6ce0e4078ec12e8fb1e8bc2aafdd04dfec520 |
| tt-forge-models | b0d3597869efbe44e58af8c1085e1de8dae9b7d2 |
