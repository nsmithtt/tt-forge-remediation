# Remediation Summary: flux1_gguf-pytorch-Dev_Q4_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux1_gguf/pytorch-Dev_Q4_K_S-single_device-inference]

## Result
SILICON_PASS — four loader bugs fixed; test passes on TT silicon in 639.98s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-loader-four-bugs-url-gated-config-dequantize

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Actual reproduced failure (same loader, different symptom after environment delta):
```
OSError: calcuis/flux1-gguf does not appear to have a file named resolve/main/flux1-dev-q4_k_s.gguf.
```

## Root cause
Four stacked bugs in the `flux1_gguf/pytorch` loader:

1. **Missing `requirements.txt`** — no `gguf>=0.10.0` beside `loader.py`; in environments without gguf pre-installed the ImportError fires before any download attempt.

2. **Wrong URL scheme** (`resolve/main` vs `blob/main`) — `diffusers` `_extract_repo_id_and_weights_name` strips only `blob/main/` from URLs (regex: `(?:blob/main/)?`). A URL containing `resolve/main/` is not stripped, so the entire path `resolve/main/flux1-dev-q4_k_s.gguf` becomes the "filename", which HuggingFace Hub cannot locate.

3. **Gated config** — after the URL fix, `from_single_file` calls `fetch_diffusers_config(checkpoint)` which reads the GGUF metadata and returns `black-forest-labs/FLUX.1-dev` as the canonical config source. That repo is gated/restricted. Fix: ship a minimal `config.json` in the loader directory and pass `config=_LOADER_DIR` so diffusers uses the local file instead of the gated HF repo.

4. **`GGUFParameter.__torch_function__` recursion under TorchDynamo** — `GGUFQuantizationConfig` leaves model weights as `GGUFParameter` subclasses. When TorchDynamo traces `dequantize_gguf_tensor → as_tensor → _make_subclass`, this triggers `__torch_function__` which calls `super().__torch_function__` again infinitely. Fix: call `_dequantize_gguf_and_restore_linear(model)` after load to convert all `GGUFParameter` weights to plain `nn.Parameter` tensors, then `torch.nn.Module.to(model, dtype)` directly (bypassing the diffusers `to()` guard that rejects dtype casting on quantized models).

## Fix
All changes in `tt_forge_models` on branch `remediation/flux1_gguf-pytorch-Dev_Q4_K_S-single_device-inference`:

- `flux1_gguf/pytorch/requirements.txt` — added `gguf>=0.10.0`
- `flux1_gguf/pytorch/loader.py`:
  - URL: `resolve/main` → `blob/main`
  - Import `_dequantize_gguf_and_restore_linear` from `diffusers.quantizers.gguf.utils`
  - Pass `config=_LOADER_DIR` to `from_single_file` to use local config
  - Call `_dequantize_gguf_and_restore_linear(self.transformer)` after load
  - Call `torch.nn.Module.to(self.transformer, compute_dtype)` to cast dtype
- `flux1_gguf/pytorch/config.json` — minimal FLUX.1-dev transformer config (guidance_embeds=true, 19 double + 38 single blocks)

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 639.98s (0:10:39)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/flux1_gguf/pytorch/loader.py`
- `tt_forge_models/flux1_gguf/pytorch/requirements.txt` (new)
- `tt_forge_models/flux1_gguf/pytorch/config.json` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355f52cb0cb7fe15c70dc8e1e4acbf29a |
| tt-mlir         | 553c0632be84a9a32e3ccf56cd8f09bd70e52d04 |
| tt-xla          | 94362e631b4e7b58bf1a60fdb34de29b7fc7b0bc |
| tt-forge-models | 2b15b209cf6fb406eb6e203e9e15c34055ca3dda |
