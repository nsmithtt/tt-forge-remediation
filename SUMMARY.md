# Remediation Summary: mixtral_gguf-causal_lm-pytorch-8x7B_Instruct_v0.1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mixtral_gguf/causal_lm/pytorch-8x7B_Instruct_v0.1_GGUF-single_device-inference]

## Result
XFAIL — Mixtral 8x7B (46.7B params, ~93 GB BF16) exceeds single-device DRAM; three loader bugs fixed before hardware ceiling reached

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.12504442074831473. Required: pcc=0.95.

## Root cause
Three compounding loader bugs prevented correct loading of the Mixtral 8x7B GGUF.

**Bug 1 — co-collected GGUF loaders poisoned load_gguf_checkpoint (TypeError on current branch)**

When pytest collects all tests, 26 other GGUF loaders (Qwen3.5, GPT-OSS-Swallow, etc.)
are imported before `mixtral_gguf`. Each of these loaders patches
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module-import time with
a wrapper `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that does not
accept the `model_to_load=` keyword argument added in transformers 5.2. When `mixtral_gguf`
calls `AutoModelForCausalLM.from_pretrained`, transformers invokes
`load_gguf_checkpoint(..., model_to_load=dummy_model)`, which reaches one of the old-signature
wrappers and raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword
argument 'model_to_load'`.

**Bug 2 — GGUF architecture misidentification (original CI failure, PCC=0.125)**

TheBloke's Mixtral-8x7B-Instruct-v0.1-GGUF stores `general.architecture = "llama"` (old
llama.cpp convention for Mixtral models). Transformers 5.x's `load_gguf_checkpoint` maps
this to `LlamaConfig`, silently dropping the `num_local_experts` and all MoE fields.
`AutoModelForCausalLM` therefore instantiates `LlamaForCausalLM` (no expert modules). The
GGUF file stores expert weights as `blk.{bid}.ffn_{gate,down,up}.{eid}.weight` — these
keys do not match `LlamaForCausalLM`'s state dict, so they are discarded. All FFN outputs
in the loaded model are random → PCC ≈ 0.125.

Additionally, `get_gguf_hf_weights_map` raises `NotImplementedError` for `model_type="mixtral"`,
and transformers has no `TensorProcessor` for the old per-expert tensor format.

**Hardware ceiling (reason for XFAIL)**

After fixing both loader bugs, Mixtral 8x7B contains ~46.7B parameters. The GGUF file is
dequantized to BF16 at load time yielding ~93 GB, which far exceeds the TT single-device
DRAM (Blackhole P150B: ~24 GB). The device raises `RuntimeError: Bad StatusOr access:
INTERNAL: Error code: 13` when the compiled graph tries to allocate device buffers.

## Fix

**Fix 1 — kwargs passthrough in 26 GGUF loaders** (commits `edba9dfd1a`, `0479ce9268` on
`remediation/mixtral_gguf-*` in `tt-forge-models`):

26 loaders had `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without
`**kwargs`. Changed all 26 to `(gguf_path, return_tensors=False, **kwargs)` and added
`**kwargs` to the internal `_orig_load_gguf_checkpoint(...)` call. This allows the
`model_to_load=` kwarg to propagate through the full chained-patch stack to the real
transformers function. Files: all 26 loader.py files in the affected model directories.

**Fix 2 — MixtralConfig + MixtralTensorProcessor in mixtral_gguf loader** (commit
`5a1f66f803` on `remediation/mixtral_gguf-*` in `tt-forge-models`):

Added to `mixtral_gguf/causal_lm/pytorch/loader.py`:
- `_patch_transformers_mixtral_gguf()` called at module import: registers
  `MixtralTensorProcessor(LlamaTensorProcessor)` in `TENSOR_PROCESSORS["mixtral"]`, which
  maps old-format per-expert tensors (`blk.{bid}.ffn_{gate,down,up}.{eid}.weight`) to
  `model.layers.{bid}.block_sparse_moe.experts.{eid}.{w1,w2,w3}.weight`. Also patches
  `get_gguf_hf_weights_map` to remap `model_type="mixtral"` → `"llama"` for the shared
  non-expert weight mapping, and wraps `load_gguf_checkpoint` to activate the
  `MixtralTensorProcessor` only when `model.config.model_type == "mixtral"`.
- `_build_mixtral_config_from_gguf()`: reads `llama.expert_count`, `llama.embedding_length`,
  `llama.block_count`, etc. from the GGUF metadata via `GGUFReader` and constructs an
  explicit `MixtralConfig`, bypassing the incorrect LlamaConfig mapping.
- `load_model()` updated to pass `config=mixtral_config` to `from_pretrained` and set
  `model.config._experts_implementation = "batched_mm"` post-load (avoids the dynamic
  `nonzero()` for-loop in eager MoE that XLA cannot trace).

**XFAIL entry** (commit `8435ac6a3` / `7868a97c6` on `remediation/mixtral_gguf-*` in
`tt-xla`):

Added `mixtral_gguf/causal_lm/pytorch-8x7B_Instruct_v0.1_GGUF-single_device-inference:
status: KNOWN_FAILURE_XFAIL` to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    1210.53s (0:20:10)
- Tier A attempts: N/A

## Files changed
- `mixtral_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)
- 26 GGUF loader files with old `_patched_load_gguf_checkpoint` signature (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7868a97c6372bf626b44047088e2ea632244dbbe |
| tt-forge-models | 5a1f66f8039ab2759ca1a456627911c6da89d0b7 |
