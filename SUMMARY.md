# Remediation Summary: joycaption-pytorch-Beta_One_HF_LLaVA-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[joycaption/pytorch-Beta_One_HF_LLaVA-single_device-inference]

## Result
XFAIL — OOM: LLaVA (Llama-3.1-8B + SigLIP) at bf16 exceeds single-device DRAM capacity after compilation overhead

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
dram-oom-single-device-llava-8b-bf16

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_FATAL: Out of Memory: Not enough space to allocate 13002342400 B DRAM buffer across 8 banks,
where each bank needs to store 1625292800 B, but bank size is 4273390016 B
(allocated: 3692802880 B, free: 580587136 B, largest free block: 544452224 B)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Three loader bugs were fixed before the hardware-class OOM was reached:

1. **`apply_chat_template` content format** (loader): The Jinja chat template for
   `fancyfeast/llama-joycaption-beta-one-hf-llava` calls `message['content'].replace()`
   at line 32, but `load_inputs` passed content as a list of multimodal dicts
   (`[{"type": "image"}, {"type": "text", ...}]`). Fixed by passing a plain string;
   the template itself prepends the image-token markers.

2. **`load_dataset` / spacy collision** (loader): `load_dataset("huggingface/cats-image")`
   triggers a dill serialization path that checks `spacy.Language`, which fails because
   the `tt_forge_models/spacy/` namespace package pollutes `sys.modules`. Fixed by
   replacing with `PIL.Image.new()` to synthesize a dummy image.

3. **`use_fast=False` on processor** (loader): The fast `SiglipImageProcessor` dispatches
   lanczos interpolation through `torch.nn.functional.interpolate`, which does not support
   lanczos — the slow PIL-based processor handles it correctly. Fixed in `_load_processor`.

After all three loader fixes the test proceeds to compilation and hits an OOM: the
`fancyfeast/llama-joycaption-beta-one-hf-llava` model (Llama-3.1-8B text decoder +
SigLIP vision encoder) at bf16 is approximately 17 GB of weights. On a single p150b
device with 8 DRAM banks of 4.27 GB each (≈34 GB total), the compilation materializes
roughly 30 GB of tensors before the OOM, leaving only 580 MB free per bank when 1.5 GB
per bank is still needed. This is a genuine hardware-capacity ceiling, not a compiler bug.

## Fix
**Loader fixes** in `tt_forge_models` (`joycaption/pytorch/loader.py`):
- `joycaption: add compressed-tensors to requirements.txt`
- `joycaption: fix apply_chat_template content format - use string not list`
- `joycaption: use_fast=False on processor to avoid lanczos interpolation error`
- `joycaption: replace load_dataset with PIL.Image to avoid spacy namespace collision`

**Test config** in `tt-xla` (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `joycaption/pytorch-Beta_One_HF_LLaVA-single_device-inference: KNOWN_FAILURE_XFAIL`
  with OOM reason explaining hardware capacity ceiling.

## Verification
- pytest exit: XFAIL (exit code 0)
- Hardware:    blackhole-p150b
- Duration:    145.07s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/joycaption/pytorch/loader.py`
- `tt_forge_models/joycaption/pytorch/requirements.txt` (new)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7816f90c27c8a7b1c1199af7d2c791f8ae25fc35 |
| tt-forge-models | 02790fc10321c10a71ee919511e4d43a378dbf36 |
