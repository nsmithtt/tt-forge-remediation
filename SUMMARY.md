# Remediation Summary: god_slayer_krix_12b_gguf-causal_lm-pytorch-GodSlayerKrix_12B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[god_slayer_krix_12b_gguf/causal_lm/pytorch-GodSlayerKrix_12B_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-missing-requirements-txt

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
The `god_slayer_krix_12b_gguf/causal_lm/pytorch/` loader directory had no
`requirements.txt`. In CI environments where `gguf` is not pre-installed in
the golden venv state, the RequirementsManager installs nothing before running
the test. When `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` is
called, transformers dispatches to `load_gguf_checkpoint`, which calls
`is_gguf_available()` first. With `gguf` absent, `is_gguf_available()` returns
False and raises ImportError. The model is a standard Llama/Mistral-based GGUF
(Q4_K_M quantization of `mradermacher/GodSlayerKrix-12B-GGUF`) and needs no
custom GGUF architecture registration.

## Fix
Added `god_slayer_krix_12b_gguf/causal_lm/pytorch/requirements.txt` containing
`gguf>=0.10.0`. This ensures the RequirementsManager installs the gguf package
before the test runs, regardless of whether it is pre-installed in the
environment.

Repository: tenstorrent/tt-forge-models
Branch: remediation/god_slayer_krix_12b_gguf-causal_lm-pytorch-GodSlayerKrix_12B_GGUF-single_device-inference
Commit: e2e2991d72

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    509.72s (0:08:29)
- Tier A attempts: N/A

## Files changed
- `god_slayer_krix_12b_gguf/causal_lm/pytorch/requirements.txt` (created)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aeecbbc5db6834c15366c1b3e5147777fca8f4d2 |
| tt-forge-models | e2e2991d72fbcd5911a154434b16fb5fd01f2187 |
