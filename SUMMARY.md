# Remediation Summary: mozilla_ai_meta_llama_3_1_70b_instruct_llamafile-causal_lm-pytorch-Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch-Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile-single_device-inference]

## Result
XFAIL — 70B Q4_K_M llamafile embeds a 42.52 GB GGUF which exceeds p150b 32 GB DRAM; loader bug (llamafile ZIP-embedded GGUF format) also fixed

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
llamafile-zip-embedded-gguf-offset-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF magic invalid
```
(The originally reported failure `ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.` was from an earlier run where gguf was not installed; with gguf 0.18.0 installed the test proceeded further and hit the format bug above.)

## Root cause
**Loader bug (llamafile format):** A `.llamafile` is a cosmopolitan APE (Actually Portable Executable) that also serves as a valid ZIP archive. The GGUF model weights are stored as a stored (uncompressed) ZIP entry at byte offset 242,745,344 inside the 42.76 GB file. Upstream `GGUFReader.__init__` calls `np.memmap(path)` starting at offset 0, which reads the MZ/PE executable header — not GGUF magic — and raises `ValueError: GGUF magic invalid`.

**Hardware capacity:** The embedded GGUF is 42,520,398,016 bytes (42.52 GB). The p150b Blackhole device has 32 GB DRAM. Even after the loader fix allows tokenizer and config to load, loading the full model weights will OOM.

## Fix
**Loader fix** in `tt_forge_models/mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch/loader.py`:

Added two functions at module level (executed at import time):

1. `_get_gguf_offset_in_llamafile(path)` — opens the file as a Python `zipfile.ZipFile`, finds the entry ending in `.gguf`, reads the local file header at `info.header_offset` to extract `fname_len` and `extra_len`, and returns `header_offset + 30 + fname_len + extra_len` as the byte offset of the GGUF data.

2. `_patch_gguf_reader_for_llamafile()` — monkey-patches `gguf.gguf_reader.GGUFReader.__init__` so that when the file is a llamafile (non-zero offset returned), it temporarily replaces `gguf_reader.np` with a shim that injects `offset=gguf_offset` into every `np.memmap` call. The original `__init__` then runs unmodified and reads GGUF magic/version/fields correctly from the right position.

**Test config XFAIL** in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`:

Added:
```yaml
  mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch-Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "Hardware capacity: 70B Q4_K_M llamafile embeds a 42.52 GB GGUF which exceeds p150b 32 GB DRAM"
```

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    blackhole-p150b
- Duration:    126.57s (test collection + config load via patched GGUFReader, stopped before full model weights load)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch/loader.py` — llamafile ZIP-embedded GGUF offset fix
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL for hardware capacity

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0f871783efda8d64dfc7e483a8822c6bde128734 |
| tt-forge-models | 41839c3044a1fd54cd2db5499eb1ada83e531716 |
