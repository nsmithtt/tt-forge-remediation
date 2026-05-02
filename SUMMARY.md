# Remediation Summary: mradermacher_qwen3_5_4b_gabliterated_gguf-causal_lm-pytorch-4B_gabliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch-4B_gabliterated_GGUF-single_device-inference]

## Result
FAIL — Tier B: qwen35 GGUF GDA/SSM layer tensor mapping missing in transformers; `Qwen3_5TextModel` exists but no GGUF weight name translation for `attn_qkv`, `attn_gate`, `ssm_*` tensors

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-gla-ssm-gguf-tensor-mapping-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The test surfaced two sequential loader failures:

**Bug 1 (fixed):**
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

**Bug 2 (terminal, Tier B):**
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.q_proj.weight | MISMATCH
  ckpt: torch.Size([8192, 2560]) vs model: torch.Size([2048, 2560])
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.k_proj.weight | MISMATCH
  ckpt: torch.Size([1024, 2560]) vs model: torch.Size([512, 2560])
```

## Root cause

**Bug 1 — loader (fixed):**
`transformers 5.2.0` added a `model_to_load` keyword argument to `load_gguf_checkpoint`. The
`_patched_load_gguf_checkpoint` wrapper in this loader (and in 24 sibling qwen35 loaders that all
overwrite the same `_gguf_utils.load_gguf_checkpoint` module attribute at import time) used a narrow
signature `(gguf_path, return_tensors=False)`. Whichever narrow-sig loader was imported last during
pytest collection would clobber the others, leaving a narrow-sig version in place at runtime.

**Bug 2 — loader, Tier B:**
`mradermacher/qwen3.5-4b-gabliterated-GGUF` is a **hybrid GDA+full-attention model** (Qwen3.5
architecture with `full_attention_interval=4`). The GGUF metadata reports:
- `qwen35.attention.head_count: 16, attention.head_count_kv: 4` (GDA layer counts)
- `qwen35.full_attention_interval: 4`

Layers 0,1,2,4,5,6,8,... (GDA) store: `attn_qkv.weight [2560,8192]`, `attn_gate.weight`,
`ssm_a`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_conv1d.weight`, `ssm_dt.bias`, `ssm_norm.weight`,
`ssm_out.weight`.

Layers 3,7,11,...,31 (full attention) store: `attn_q.weight [2560,8192]`, `attn_k.weight [2560,1024]`,
`attn_v.weight [2560,1024]`, `attn_output.weight [4096,2560]`.

The loader maps `qwen35 → qwen3` (Qwen3ForCausalLM), which expects **all** layers to have `q_proj=[2048,2560]`
(16 heads × 128 head_dim). The full-attention layers in the GGUF use `head_dim=256` and complex-RoPE
encoding (`q_proj=[16*256*2, hidden]=[8192, 2560]`), causing size mismatches.

`transformers 5.2.0` does have `Qwen3_5ForCausalLM` / `Qwen3_5TextModel` (model_type `qwen3_5_text`)
which supports `layer_types=["linear_attention","full_attention",...]`, but there is **no GGUF tensor
name mapping** for `qwen3_5_text` / `qwen35` in `GGUF_TO_TRANSFORMERS_MAPPING`. The GDA-layer tensors
(`attn_qkv`, `attn_gate`, `ssm_*`) have no translations to HuggingFace weight names, so loading via
GGUF is impossible without a new mapping table.

## Fix

**Bug 1 (done in tt_forge_models):**
Changed all 26 affected loaders (25 with narrow-sig + 1 already had a first commit):
```python
# Before
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

Files changed: `mradermacher_qwen3_5_4b_gabliterated_gguf`, `mradermacher_qwen3_5_4b_unfiltered_gguf`,
`mradermacher_qwen3_5_4b_unredacted_max_gguf`, `mradermacher_qwen3_5_4b_abliterated_i1_gguf`,
`mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`,
`mradermacher_bartleby_qwen3_5_4b_gguf`, `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`,
`mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`,
`mradermacher_qwen3_5_27b_homebrew_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`,
`mradermacher_luna_qwen3_5_27b_v5_i1_gguf`,
`mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`,
`mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_vilm_0_8b_sft_gguf`,
`tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`,
`daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `qwen_3_5_imatrix_gguf`,
`unified_reward_flex_qwen35_27b_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`,
`gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`
(all in `causal_lm/pytorch/loader.py`)

**Bug 2 (proposed fix for future work):**
Add `qwen3_5_text` / `qwen35` entry to `transformers.modeling_gguf_pytorch_utils.GGUF_TO_TRANSFORMERS_MAPPING`
with tensor name translations for both GDA layers (`attn_qkv` → packed QKV split, `attn_gate`,
`ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt`, `ssm_norm`, `ssm_out`) and full-attention
layers (`attn_q`, `attn_k`, `attn_v`, `attn_output`). Also map `full_attention_interval` GGUF metadata
field → `layer_types` list in `Qwen3_5TextConfig`. This is a ~100-line addition to the transformers
upstream, likely as a PR to `huggingface/transformers`.

## Tier B justification
**new-infrastructure**: No GGUF weight-name translation exists for the Qwen3.5 hybrid GDA+full-attention
architecture. The `Qwen3_5TextModel` class exists in transformers 5.2.0, but `GGUF_TO_TRANSFORMERS_MAPPING`
has no `qwen35` or `qwen3_5_text` entries, and the GDA tensor names (`attn_qkv`, `ssm_*`) have no
HuggingFace equivalents in any existing mapping. Implementing this requires new infrastructure in the
transformers GGUF loader, not a scoped fix in tt_forge_models or the tt compiler stack.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    356.43s (second run after Bug 1 fix)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py` (Bug 1)
- 24 additional qwen35/gpt-oss GGUF loaders with narrow-sig patch (Bug 1, session contamination)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f21a530d0eb1eca97a8732fc8b14d2c81764bbf0 |
| tt-forge-models | 959e56c23c647e6cb110568d64d0bded17a9045d |
