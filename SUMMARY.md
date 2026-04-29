# Remediation Summary: crow_9b_heretic_i1_gguf-causal_lm-pytorch-9B_HERETIC_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[crow_9b_heretic_i1_gguf/causal_lm/pytorch-9B_HERETIC_I1_GGUF-single_device-inference]

## Result
XFAIL — 9B model with 248K-token vocabulary dequantized to bfloat16 exhausts single-device DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
oom-9b-extended-vocab-bf16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 4068474880 B DRAM buffer across 8 banks,
where each bank needs to store 508559360 B, but bank size is 4273390016 B
(allocated: 4230746624 B, free: 42643392 B, largest free block: 14712832 B)
```

The OOM occurs during `prepareInputTensor` → `toLayout` → `tilize` when sending the
embedding/lm_head weight tensors to device for the first inference pass.

## Root cause

Two loader bugs had to be fixed before reaching the hardware limit:

**Bug 1 (loader)**: During pytest collection, 26 qwen35 GGUF loaders monkey-patched
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a wrapper that
lacked the `model_to_load` keyword argument added in transformers 5.x. Because
`discover_loader_paths()` imports every loader at collection time, this broken patch
was installed globally, causing `TypeError: unexpected keyword argument 'model_to_load'`
even for the crow test which does not patch anything itself.

**Bug 2 (loader)**: The crow loader used the `qwen35.vocab_size` GGUF metadata field
(which reports the base Qwen3 vocabulary size 151936) instead of the actual embedding
matrix dimensions. Crow-9B-HERETIC has an extended vocabulary of 248320 tokens, so
`Qwen3NextConfig` was built with the wrong vocab size, causing a shape mismatch in
`load_state_dict` for `embed_tokens.weight` and `lm_head.weight`.

After both loader bugs were fixed, the model loads and compiles successfully, but
fails at runtime due to hardware capacity:

- **Model size in bfloat16**: 9B parameters × 2 bytes = ~18 GiB of weights alone.
- **Extended vocabulary overhead**: embed_tokens (248320 × 4096 × 2 = 1.94 GiB) +
  lm_head (same) = 3.78 GiB for embeddings alone, on top of transformer layer weights.
- **Device DRAM**: 8 banks × ~3.98 GiB/bank ≈ 31.8 GiB total.
- **Observed at failure**: 3.94 GiB/bank allocated (31.5 GiB total), only 40 MiB/bank
  free. The next allocation (508 MiB/bank for the embedding tilize) cannot be satisfied.

The Q4_K_M GGUF file is ~5 GiB compressed. After dequantization to bfloat16 for TTNN,
the model expands to ~18 GiB. This exceeds what is available after the runtime reserves
its working space on a single-device configuration.

## Fix

**In tt_forge_models** (`remediation/crow_9b_heretic_i1_gguf-…` branch):

1. `crow_9b_heretic_i1_gguf/causal_lm/pytorch/loader.py` — Completely rewritten to
   directly instantiate `Qwen3NextForCausalLM` (the correct HF class for the qwen35
   hybrid SSM + full-attention architecture) by reading GGUF fields for config and
   manually mapping GGUF tensor names to HF parameter names. The GGUF metadata
   `qwen35.vocab_size` is overridden with the actual embedding tensor shape
   (stored as `[hidden_size, vocab_size]` in GGUF, so `tensor.shape[-1]` gives 248320).

2. 26 other qwen35 GGUF loaders — Added `model_to_load=None` to the
   `_patched_load_gguf_checkpoint` signature and forwarded it to the original function,
   fixing the global-pollution TypeError during pytest collection.

**In tt-xla** (`remediation/crow_9b_heretic_i1_gguf-…` branch):

3. `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — Added
   `KNOWN_FAILURE_XFAIL` entry for the crow test with the OOM reason string.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

N/A — disposition is XFAIL (hardware-class), not FAIL.

## Verification
- pytest exit: FAIL (OOM on silicon, as expected for XFAIL)
- Hardware:    wormhole (8-bank device, ~32 GiB DRAM)
- Duration:    1628.92s (0:27:08) — time to compile + OOM at execution
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)
- `tt_forge_models/crow_9b_heretic_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/{26 qwen35 loaders}/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d35295e80d7bf67bcab0aecb46e1ae152b5caefa |
| tt-forge-models | 07d06b26733173493889b0007ff9e00c0040b6c4 |
