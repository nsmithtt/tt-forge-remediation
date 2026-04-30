# Remediation Summary: glm_4_7_flash_gguf-causal_lm-pytorch-4.7_Flash_ngxson_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_gguf/causal_lm/pytorch-4.7_Flash_ngxson_GGUF-single_device-inference]

## Result
FAIL — deepseek2 GGUF weight mapping absent from transformers; three weight-loading bugs remain after partial loader fixes

## Stack layer
loader

## Tier
B

## Bug fingerprint
deepseek2-gguf-tensor-processor-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure:
```
raise AttributeError(
```

Reproduced locally as:
```
KeyError: 'deepseek_v2'
```
in `transformers/integrations/ggml.py` `convert_gguf_tokenizer`, because `GGUF_TO_FAST_CONVERTERS`
had no entry for `"deepseek_v2"`.

After applying the tokenizer fix and the `validate_architecture` patch (commit `2b2f3a3cd6`), a new
failure surfaces:

```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

With the transformers LOAD REPORT showing:
```
model.layers.{1...46}.mlp.experts.gate_up_proj   | MISSING  |
model.layers.{0...46}.self_attn.kv_b_proj.weight | MISSING  |
model.layers.{1...46}.mlp.experts.down_proj      | MISSING  |
model.layers.{0...46}.self_attn.q_b_proj.weight  | MISMATCH | ckpt: torch.Size([5120, 768]) vs model: torch.Size([6400, 768])
```

## Root cause
Two bugs were fixed in commit `2b2f3a3cd6`:

**Bug 1 (fixed):** `GGUF_TO_FAST_CONVERTERS` lacked a `"deepseek_v2"` entry. During pytest
collection, `qwen3_32b_vl_glm_4_7_flash_hi16_heretic_uncensored_thinking_i1_gguf` and similar
loaders patch `tokenization_utils_tokenizers.load_gguf_checkpoint` and chain through GLM's
`deepseek2→deepseek_v2` model_type remap. The downstream `convert_gguf_tokenizer` call then
receives `architecture="deepseek_v2"` but only `"deepseek2"` was in `GGUF_TO_FAST_CONVERTERS`,
causing the `KeyError`.

**Bug 2 (fixed):** `DeepseekV2Config.validate_architecture` checks
`hidden_size % num_attention_heads != 0`. GLM-4.7-Flash has `hidden_size=2048`,
`num_attention_heads=20` → `2048 % 20 = 8 ≠ 0`. For MLA configs (`q_lora_rank is not None`),
this check is irrelevant because the head count never divides hidden_size directly; the fix
patches `__class_validators__` to skip the check when `q_lora_rank is not None`.

**Bug 3 (Tier B, unfixed):** The GLM-4.7-Flash GGUF (ngxson's `GLM-4.7-Flash-Q4_K_M.gguf`)
was created with llama.cpp's deepseek2 tensor naming, but transformers has no
`TensorProcessor` registered for `deepseek2` or `deepseek_v2` in its `TENSOR_PROCESSORS`
dict. As a result three classes of weights fail to load:

1. **MoE expert weights (MISSING):** The GGUF stores expert tensors as 3-D arrays with the
   expert index as the *last* axis (`blk.N.ffn_gate_exps [2048,1536,64]`,
   `blk.N.ffn_up_exps [2048,1536,64]`, `blk.N.ffn_down_exps [1536,2048,64]`). The
   deepseek2 `get_tensor_name_map` maps `mlp.experts.gate_up_proj` →
   `blk.N.ffn_gate_up_exps` (combined), but the GGUF file stores them separately as
   `ffn_gate_exps` and `ffn_up_exps`. Without `preprocess_name` to strip the expert index
   AND `perform_fallback_tensor_mapping` to split the mapping, both `gate_up_proj` and
   `down_proj` are not found in the GGUF and load as random noise.

2. **kv_b_proj (MISSING):** The GGUF stores the MLA KV-B projection split into
   `blk.N.attn_k_b [192,512,20]` and `blk.N.attn_v_b [512,256,20]`, but the deepseek2
   name_map maps `self_attn.kv_b_proj` → `blk.N.attn_kv_b` (combined, absent from the
   file). No TensorProcessor exists to concatenate k_b + v_b into kv_b_proj.

3. **q_b_proj (MISMATCH, architectural):** The GGUF stores `blk.N.attn_q_b [768,5120]`
   (= Linear(q_lora_rank=768, num_heads×qk_nope_head_dim=5120)), but HF's
   `DeepseekV2ForCausalLM` builds `q_b_proj` as
   Linear(768, num_heads×(qk_nope+qk_rope)=6400). The missing 1280 output rows correspond
   to the per-head rope query projection. In the original GLM/DeepSeek-V2 implementation,
   q_pe is derived by sharing the key rope position embedding (k_pe, a single
   qk_rope_head_dim=64 vector from `kv_a_proj_with_mqa`) rather than having a per-head
   projection inside q_b_proj. HF's implementation includes q_pe rows in q_b_proj, so the
   GGUF checkpoint has a structurally different q_b_proj that cannot be loaded into the HF
   model without either changing the model class or padding with zeros and sharing k_pe.

## Fix
Commit `2b2f3a3cd6` on branch
`remediation/glm_4_7_flash_gguf-causal_lm-pytorch-4.7_Flash_ngxson_GGUF-single_device-inference`
in `tt_forge_models`:

- **`glm_4_7_flash_gguf/causal_lm/pytorch/loader.py`** — `_patch_transformers_deepseek2_gguf()`:
  - Added `GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter` (Bug 1).
  - Added `DeepseekV2Config.__class_validators__` patch to skip `hidden_size %
    num_attention_heads` check when `q_lora_rank is not None` (Bug 2).

The remaining three weight-loading bugs (Bug 3) are **not fixed** and require the following
proposed work:

**Proposed fix for Bug 3a (MoE experts):** Register a `DeepseekV2TensorProcessor` in
`TENSOR_PROCESSORS["deepseek_v2"]` (in the monkey-patch) that:
- Overrides `preprocess_name` to remove expert indices (like `Qwen2MoeTensorProcessor`).
- Overrides `perform_fallback_tensor_mapping` to add `ffn_gate_exps` + `ffn_up_exps` → same
  `gate_up_proj` HF name (because deepseek2's name_map returns `ffn_gate_up_exps` which is
  absent from the file).
- Implements `process` to permute 3-D stacked expert tensors (expert dim last → first) and
  interleave gate + up into the combined `gate_up_proj`.

**Proposed fix for Bug 3b (kv_b_proj):** In the same `DeepseekV2TensorProcessor`, when
`attn_k_b` or `attn_v_b` is encountered, store partial results and concatenate into
`kv_b_proj` once both are available.

**Proposed fix for Bug 3c (q_b_proj architecture):** Patch the model's forward method (or
create a custom model subclass) so that q_pe is derived from k_pe (broadcast across heads)
rather than from rows of q_b_proj. This allows loading the GGUF's 5120-row `attn_q_b`
unchanged while correctly computing the decoupled-RoPE query positions.

## Tier B justification
The deepseek2 GGUF weight mapping is entirely absent from transformers — no `TensorProcessor`
for `deepseek2`/`deepseek_v2` exists. Implementing the full mapping requires:
- A new `DeepseekV2TensorProcessor` class (~150+ lines).
- Patching or extending `get_gguf_hf_weights_map` to substitute the name-map entries for
  combined `ffn_gate_up_exps` / `attn_kv_b` with the GGUF file's separate tensors.
- For q_b_proj: either a custom model subclass or a runtime forward-method patch to share
  k_pe as q_pe (architectural difference between GGUF and HF implementations).

This constitutes new infrastructure (`new-infrastructure`) that spans new class implementation,
transformer internals patching, and a potential model architecture change — well beyond a
single-function scoped fix.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 2280.45s (0:38:00) — model loaded but failed at weight validation
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/glm_4_7_flash_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 6f0f1a1e93c5dc790918ea9361381650a981191f |
| tt-forge-models | 2b2f3a3cd6521be0f0b0847b3416230ac55fb964 |
