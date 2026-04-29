# Remediation Summary: chandra_ocr_2_gguf-causal_lm-pytorch-chandra_ocr_2_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chandra_ocr_2_gguf/causal_lm/pytorch-chandra_ocr_2_GGUF-single_device-inference]

## Result
FAIL — Chandra OCR 2 uses a Mamba2+Attention hybrid architecture (SSM layers + full-attention layers) not supported by transformers' `qwen35` GGUF loader; requires implementing a new hybrid model class

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-ssm-hybrid-arch-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Three bugs were encountered in sequence.

**Bug 1 (fixed):** `OSError: prithivMLmods/chandra-ocr-2-GGUF does not appear to have a file named chandra-ocr-2-Q4_K_M.gguf`

The loader specified `GGUF_FILE = "chandra-ocr-2-Q4_K_M.gguf"` but the HuggingFace repo
`prithivMLmods/chandra-ocr-2-GGUF` has no Q4_K_M quantization. Available files are
`chandra-ocr-2.BF16.gguf`, `chandra-ocr-2.F16.gguf`, `chandra-ocr-2.F32.gguf`,
`chandra-ocr-2.Q8_0.gguf`, and `mmproj-*` variants. Fix: use `chandra-ocr-2.Q8_0.gguf`.

**Bug 2 (fixed):** `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

26 GGUF loaders in tt_forge_models monkey-patch `load_gguf_checkpoint` at module level
with a fixed signature `(gguf_path, return_tensors=False)`. Transformers 5.x added
`model_to_load=None` to this signature; when any of these loaders is imported in the same
pytest session, the patched function receives the new kwarg and raises `TypeError`. Fix:
update all 26 loaders to accept `**kwargs` and forward to `_orig_load_gguf_checkpoint`.

**Bug 3 (unfixed):** `RuntimeError: You set 'ignore_mismatched_sizes' to 'False', thus raising an error.`

After applying both fixes, `AutoModelForCausalLM.from_pretrained` raises a size-mismatch
error on the full-attention layers:

```
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH |
  ckpt: torch.Size([8192, 2560]) vs model: torch.Size([2048, 2560])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH |
  ckpt: torch.Size([1024, 2560]) vs model: torch.Size([512, 2560])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_norm.weight | MISMATCH |
  ckpt: torch.Size([256]) vs model: torch.Size([128])
```

## Root cause
Chandra OCR 2 (`prithivMLmods/chandra-ocr-2-GGUF`) is a Mamba2+Transformer hybrid model
based on Qwen3.5-4.8B. The GGUF metadata confirms this via `qwen35.ssm.*` fields:
- `qwen35.ssm.conv_kernel = 4`
- `qwen35.ssm.group_count = 16`
- `qwen35.ssm.inner_size = 4096`
- `qwen35.ssm.state_size = 128`
- `qwen35.ssm.time_step_rank = 32`
- `qwen35.full_attention_interval = 4`

With `block_count=32` and `full_attention_interval=4`, the model has 8 full-attention layers
(at positions 3, 7, 11, 15, 19, 23, 27, 31) each using `head_dim=256` and 24 SSM-hybrid
layers. The `qwen35` GGUF architecture is mapped to standard `Qwen3Config` by the patching
code, which assumes uniform `num_attention_heads=16, head_dim=128` across all layers —
causing the 4× size mismatch in q/k projections and norms on the 8 full-attention layers.
This is identical to the Aethon-4b (Featherlabs/Aethon-4b) failure (same architecture class).

## Fix
Two bugs fixed in `tt-xla/third_party/tt_forge_models` on branch
`remediation/chandra_ocr_2_gguf-causal_lm-pytorch-chandra_ocr_2_GGUF-single_device-inference`:

1. `chandra_ocr_2_gguf/causal_lm/pytorch/loader.py`: Changed `GGUF_FILE` from
   `chandra-ocr-2-Q4_K_M.gguf` to `chandra-ocr-2.Q8_0.gguf`.

2. 26 loader files: Updated `_patched_load_gguf_checkpoint` signature from
   `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, **kwargs)`
   and updated the call to `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`.

For the unfixed SSM hybrid architecture bug: the proposed fix is to implement a new
`Qwen35HybridForCausalLM` PyTorch class in the loader that correctly handles both SSM layers
(with Mamba2 selective scan) and full-attention layers (with larger head_dim=256). However,
implementing Mamba2 SSM ops (`selective_scan`) is also required for TT hardware, making this
a cross-repo, new-infrastructure effort.

## Tier B justification
The SSM hybrid architecture bug requires new-infrastructure: implementing a new
`Qwen35HybridForCausalLM` PyTorch class plus Mamba2 `selective_scan` op support in
tt-mlir/tt-metal. Touches more than 3 files across multiple repos, and Mamba2 SSM kernels
do not exist in the TT compiler stack.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 340.84s (0:05:40) to reach SSM architecture failure
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/chandra_ocr_2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 40752bfbbb1ddd90cf7878f6ffa291687f34a84b |
| tt-forge-models | ae55867f9448a9b81a335173412d96fd0aef4068 |
