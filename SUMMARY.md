# Remediation Summary: cosmos_reason1_gguf-causal_lm-pytorch-7B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cosmos_reason1_gguf/causal_lm/pytorch-7B_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-qwen2vl-architecture-not-in-transformers-supported-list

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: GGUF model with architecture qwen2vl is not supported yet.

(The originally reported error was `ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.` — this was already fixed by gguf>=0.10.0 being installed, and the reproduction revealed the next-in-line error: qwen2vl not in GGUF_SUPPORTED_ARCHITECTURES.)

## Root cause
Loader layer bug. `transformers 5.2.0` does not include `qwen2vl` in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING["config"]`. The Cosmos-Reason1-7B GGUF file (from `unsloth/Cosmos-Reason1-7B-GGUF`) reports `general.architecture = 'qwen2vl'`, which causes `load_gguf_checkpoint` to raise `ValueError` before any model weights are read.

The GGUF file contains only language model weights (`token_embd`, `output`, `output_norm`, `blk.*`) — no vision encoder. The gguf library already knows the `QWEN2VL` architecture (`MODEL_ARCH.QWEN2VL`) with the same tensor naming convention as `QWEN2`. The fix adds `qwen2vl` to the transformers GGUF config mapping and reinterprets the loaded model as `qwen2` (with `attention_bias=True` to capture the Q/K/V biases present in Qwen2-VL attention layers) so `AutoModelForCausalLM` instantiates `Qwen2ForCausalLM` with the correct weights.

## Fix
`cosmos_reason1_gguf/causal_lm/pytorch/loader.py` in `tt-forge-models` (commit `888540bde0`):

Added `_patch_transformers_qwen2vl_gguf()` called at import time, which:
1. Appends `"qwen2vl"` to `GGUF_SUPPORTED_ARCHITECTURES`
2. Adds a `"qwen2vl"` entry to `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with the same field mapping as `"qwen2"` (the GGUF metadata fields are identical)
3. Registers `GGUFQwen2Converter` for the `qwen2vl` tokenizer type (same BPE tokenizer)
4. Wraps `load_gguf_checkpoint` to rewrite `model_type` from `qwen2vl` → `qwen2` and set `attention_bias=True` in the returned config

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    120.08s (0:02:00)
- Tier A attempts: N/A

## Files changed
- `cosmos_reason1_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1b7b73bc9e9f17432b66a1e9c5d5c27844e0efa1 |
| tt-forge-models | 888540bde062ee869b8c634d168f0f2dbc0200a5 |
