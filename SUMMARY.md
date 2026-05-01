# Remediation Summary: hypernova_60b_2602_gguf-causal_lm-pytorch-Hypernova_60B_2602_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hypernova_60b_2602_gguf/causal_lm/pytorch-Hypernova_60B_2602_Q4_K_M-single_device-inference]

## Result
XFAIL — 60B Qwen3-MoE model dequantizes to ~120 GB BF16, exceeding all single-device TT DRAM (p150b=24 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-gpt-oss-arch-not-registered, hardware-60b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise AttributeError(

Full chain: The GGUF file declares `general.architecture = gpt-oss`. Without
registering this architecture in `GGUF_SUPPORTED_ARCHITECTURES`, transformers
raises `ValueError: GGUF model with architecture gpt-oss is not supported
yet.` In a full pytest session where a `gpt_oss_swallow` loader was collected
before this test, the gpt_oss_swallow module installed its own
`_patched_load_gguf_checkpoint` with the old fixed signature
`(gguf_path, return_tensors=False)`. When `from_pretrained` later called this
wrapper with `model_to_load=model`, the kwarg was silently dropped, leaving
`model_to_load=None` inside the real `load_gguf_checkpoint`. That caused
`get_gguf_hf_weights_map(None, processor)` to access `None.config.model_type`,
which raises `AttributeError`.

## Root cause
**Loader layer.** The `hypernova_60b_2602_gguf` loader did not patch
`GGUF_SUPPORTED_ARCHITECTURES` or the four `load_gguf_checkpoint` binding
sites to register `gpt-oss` (the GGUF architecture key for this model's
`general.architecture` field) as an alias for `qwen3_moe`. The same gap
existed in the `gpt_oss_swallow` family loaders and was already fixed there;
the hypernova loader was missing the equivalent fix entirely.

The model is `Qwen3MoeForCausalLM` with 80 experts (derived from
GPT-OSS-120B). After the loader fix, the model loads into CPU RAM (38 GB
Q4_K_M GGUF dequantizes to ~120 GB BF16 in host memory) but the TT device
DRAM ceiling (p150b: 24 GB) makes single-device inference impossible.

## Fix
**tt_forge_models** (`remediation/hypernova_60b_2602_gguf-causal_lm-pytorch-Hypernova_60B_2602_Q4_K_M-single_device-inference`, commit `a4f7c84224`):

- `hypernova_60b_2602_gguf/causal_lm/pytorch/loader.py`: Added
  `_patch_gpt_oss_support()` (registers `gpt-oss` in `GGUF_SUPPORTED_ARCHITECTURES`
  and all related GGUF mapping tables as an alias for `qwen3_moe`) and
  `_patched_load_gguf_checkpoint(*args, **kwargs)` that calls the real function
  transparently and remaps `model_type='gpt-oss'` → `'qwen3_moe'` in the
  returned config. Patched all four import-site bindings at module load time.
  Also added `model.config._experts_implementation = "batched_mm"` in
  `load_model` (avoids `histc-on-Int` failure on XLA device for Qwen3Moe) and
  a `chat_template is not None` guard in `load_inputs`.

- `hypernova_60b_2602_gguf/causal_lm/pytorch/requirements.txt`: Created with
  `gguf>=0.10.0`.

**tt-xla** (`remediation/hypernova_60b_2602_gguf-causal_lm-pytorch-Hypernova_60B_2602_Q4_K_M-single_device-inference`, commit `f310bb7a0`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  Added `KNOWN_FAILURE_XFAIL` entry for the test with reason explaining the
  hardware capacity ceiling.

## Verification
- pytest exit: not-run (hardware-class XFAIL; model cannot be transferred to device)
- Hardware:    blackhole-p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hypernova_60b_2602_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/hypernova_60b_2602_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f310bb7a0a9307166484ac0cc15e5c8ea4738728 |
| tt-forge-models | a4f7c8422408e072d9cd2f6f112227ed0c7acaac |
