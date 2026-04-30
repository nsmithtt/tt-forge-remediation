# Remediation Summary: flux2_gguf-pytorch-flux2_dev_Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux2_gguf/pytorch-flux2_dev_Q8_0-single_device-inference]

## Result
XFAIL — FLUX.2 Q8_0 (32.2B parameters) exceeds single-device DRAM (~34GB) at both BF16 (~64GB) and Q8_0 (~36GB) quantization

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-parameter-torch-function-dynamo-recursion

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

## Root cause
Two loader bugs prevented the model from reaching silicon:

**Bug 1 — Gated config access:** `from_single_file` reads the GGUF tensor keys
and detects the FLUX.2 architecture via `single_stream_modulation.lin.weight`,
mapping it to `black-forest-labs/FLUX.2-dev` (a gated repo). Without HF
credentials authorizing that repo, `load_config` raises a 403 GatedRepoError
and then an OSError. The original test machine had access; ours did not, so
this loader bug was exposed. Fixed by shipping a local `config.json` with
the FLUX.2 default architecture parameters (verified against GGUF tensor
shapes: 8 double blocks, 48 single blocks, hidden_size=6144, 32 attention
heads × 128 head_dim, joint_attention_dim=15360) and passing `config=<src_dir>`
to `from_single_file`.

**Bug 2 — GGUFParameter Dynamo recursion:** After successful loading,
`GGUFParameter.__torch_function__` recurses infinitely under TorchDynamo
tracing, causing the reported InternalTorchDynamoError. Fixed by calling
`_dequantize_gguf_and_restore_linear(transformer)` to replace all GGUFLinear
layers with plain `nn.Linear` layers, then clearing `_hf_quantizer=None` and
`is_quantized=False` so `ModelMixin.to()` allows subsequent dtype casts.

**Hardware capacity ceiling (XFAIL):** After both loader bugs were fixed, the
model compiled but OOM'd on-device trying to allocate a ~648 MB buffer after
the model weights had already consumed ~93% of the 34.2 GB DRAM (8 banks ×
4.27 GB). Root cause: FLUX.2 Q8_0 has 32.2 billion parameters. At BF16
(required after dequantization), the weights require ~64 GB. Even at Q8_0
quantization, the weight data alone is ~36 GB, which already exceeds the
device's ~34 GB DRAM before compilation artifacts and runtime tensors are
accounted for. This is a genuine hardware capacity ceiling, not a compiler bug.

## Fix
**Loader fixes** (tt_forge_models, branch `remediation/flux2_gguf-pytorch-flux2_dev_Q8_0-single_device-inference`):

- `flux2_gguf/pytorch/src/config.json` — new file: FLUX.2 transformer config
  with default architecture parameters, bypassing the gated
  `black-forest-labs/FLUX.2-dev` repo.
- `flux2_gguf/pytorch/src/model_utils.py` — updated `load_flux2_gguf_transformer`:
  added `config=_CONFIG_DIR` argument to `from_single_file`; added
  `_dequantize_gguf_and_restore_linear(transformer)`;
  cleared `transformer._hf_quantizer = None` and `transformer.is_quantized = False`;
  called `transformer.to(compute_dtype)` to cast any residual float16 tensors.

**Test config** (tt-xla, branch `remediation/flux2_gguf-pytorch-flux2_dev_Q8_0-single_device-inference`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` —
  added `flux2_gguf/pytorch-flux2_dev_Q8_0-single_device-inference` with
  `status: KNOWN_FAILURE_XFAIL`.

## Verification
- pytest exit: FAIL (OOM after loader fixes — hardware-class, not compiler bug)
- Hardware: wormhole (n150, 8 banks × 4.27 GB = 34.2 GB DRAM)
- Duration: 5m 33s (OOM run with loader fixes applied)
- Tier A attempts: N/A

## Files changed
- `flux2_gguf/pytorch/src/config.json` (new, in tt_forge_models)
- `flux2_gguf/pytorch/src/model_utils.py` (modified, in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (modified, in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1dfb684be009e93c3a7e8009a9c4083a32be1a1c |
| tt-forge-models | 80c486d4e1 (remediation/flux2_gguf-pytorch-flux2_dev_Q8_0-single_device-inference) |
