# Remediation Summary: aethon_4b_i1_gguf-causal_lm-pytorch-aethon_4b_i1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aethon_4b_i1_gguf/causal_lm/pytorch-aethon_4b_i1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Aethon-4b uses a Mamba2+Attention hybrid architecture (SSM layers + full-attention layers) not supported by transformers' `qwen35` GGUF loader; requires implementing a new hybrid model class

## Stack layer
loader

## Tier
N/A

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
Two bugs were encountered in sequence.

**Bug 1 (fixed):** `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

26 GGUF loaders in tt_forge_models monkey-patch `load_gguf_checkpoint` at module level
with a fixed signature `(gguf_path, return_tensors=False)`. Transformers 5.x added
`model_to_load=None` to this signature; when any of these loaders is imported in the same
pytest session as the aethon test, the patched function receives the new kwarg and raises
`TypeError`. Fix: update all 26 loaders to accept `model_to_load=None` and forward it to
`_orig_load_gguf_checkpoint`. This fix already existed on the `remediation/aethon_4b_i1_gguf-*`
branch of tt_forge_models.

**Bug 2 (unfixed):** `RuntimeError: You set 'ignore_mismatched_sizes' to 'False'`

After applying bug 1 fix, `AutoModelForCausalLM.from_pretrained` raises a size-mismatch error:

```
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH |
  ckpt: torch.Size([8192, 2560]) vs model: torch.Size([2048, 2560])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH |
  ckpt: torch.Size([1024, 2560]) vs model: torch.Size([512, 2560])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_norm.weight | MISMATCH |
  ckpt: torch.Size([256]) vs model: torch.Size([128])
```

## Root cause
Aethon-4b (`Featherlabs/Aethon-4b`, quantized by mradermacher) is a Mamba2+Transformer hybrid
model trained on Qwen3.5-4B. The GGUF metadata confirms this via the `qwen35.ssm.*` fields
(`ssm_state_size=128`, `ssm_conv_kernel`, `ssm_group_count`, `ssm_alpha`, `ssm_beta`,
`ssm_dt`, `full_attention_interval`).

The 32-layer model has two distinct layer types:
1. **Hybrid layers (24 of 32, e.g. blk.0–2, 4–6, …):** Each has a combined `attn_qkv` linear
   attention, a Mamba2 SSM (`ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_out`), and an FFN.
   Tensor names in GGUF: `blk.N.attn_qkv.weight [2560,8192]`, `blk.N.attn_gate.weight`,
   `blk.N.ssm_a`, `blk.N.ssm_conv1d.weight [4,8192]`, etc.
2. **Full-attention layers (8 of 32, at indices 3,7,11,15,19,23,27,31 i.e. every 4th):**
   Standard GQA with separate Q/K/V projections and larger head_dim=256 (vs. 128 in hybrid
   layers). Tensor names: `blk.N.attn_q.weight [2560,8192]`, `blk.N.attn_k.weight [2560,1024]`,
   `blk.N.attn_q_norm.weight [256]`, etc.

Transformers maps the GGUF architecture key `qwen35` → `Qwen3Config` / `Qwen3ForCausalLM`.
This creates a standard Qwen3.5 model where ALL layers have identical attention configs
(`num_attention_heads=16`, `head_dim=128`). When loading the GGUF weights:
- The 24 hybrid layers load their attention weights into mismatched Qwen3 attention modules
  (no SSM sub-module exists in the Qwen3 class at all)
- The 8 full-attention layers have q_proj weight shape `[8192,2560]` (32 heads × 256 head_dim)
  but the Qwen3 model expects `[2048,2560]` (16 heads × 128 head_dim) → MISMATCH error

## Fix
Proposed fix (not implemented): implement a `Qwen35HybridForCausalLM` PyTorch class that
interleaves Mamba2 SSM blocks and standard full-attention blocks according to
`full_attention_interval`. This requires:
- A new `Qwen35HybridConfig` class with per-type layer configuration
- A `Mamba2Block` sub-module (alpha/beta projections, conv1d, dt gate, SSM state scan)
- A `FullAttentionBlock` sub-module (standard GQA with head_dim=256)
- A custom GGUF weight remapping from `blk.N.ssm_*` / `blk.N.attn_qkv` tensor names to the
  new module hierarchy
- Registration in `GGUF_SUPPORTED_ARCHITECTURES` for the `qwen35` key

Even if the loader were fixed, the Mamba2 SSM operations (`selective_scan`, state-space
convolution) have no TT hardware kernels and would likely fail at silicon level as a
subsequent Tier B compiler bug.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — this is a loader bug, not a compiler-stack bug. Tier classification is N/A.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    383.19s (after bug 1 fix; bug 2 prevents reaching silicon)
- Tier A attempts: N/A

## Files changed
`tt-xla/third_party/tt_forge_models` (remediation branch, 26 files):
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3b3e2924a37e8bc669b52ae3ff346955e8aea06d |
