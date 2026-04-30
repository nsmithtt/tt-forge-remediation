# Remediation Summary: glm_4_7_flash_gguf-causal_lm-pytorch-4.7_Flash_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_gguf/causal_lm/pytorch-4.7_Flash_GGUF-single_device-inference]

## Result
FAIL — deepseek2 GGUF weight format incompatible with HF DeepseekV2ForCausalLM; requires custom TensorProcessor (Tier B new-infrastructure)

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
Original failure: `KeyError: 'deepseek_v2'` in `convert_gguf_tokenizer` during tokenizer loading.

Three loader bugs were fixed before reaching the terminal Tier B failure:

1. `KeyError: 'deepseek_v2'` — `GGUF_TO_FAST_CONVERTERS` had `deepseek2` but not `deepseek_v2`. The `load_gguf_checkpoint` patch remaps `model_type deepseek2→deepseek_v2`; the tokenizer init looks up that remapped string.

2. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` — narrow-sig GGUF loaders imported after `glm_4_7_flash_ggml_org_gguf` (e.g. `gpt_oss_swallow_*`, `mradermacher_*`) clobber `modeling_utils.load_gguf_checkpoint` with `(gguf_path, return_tensors=False)` wrappers that drop `model_to_load`.

3. `NotImplementedError: Unknown gguf model_type: deepseek_v2 in gguf-py` — `get_gguf_hf_weights_map` looked up `deepseek_v2` in gguf-py `MODEL_ARCH_NAMES`, which only contains `"deepseek2"`.

Terminal failure after all three loader fixes:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
model.layers.{1...46}.mlp.experts.gate_up_proj   | MISSING
model.layers.{0...46}.self_attn.kv_b_proj.weight | MISSING
model.layers.{0...46}.self_attn.q_b_proj.weight  | MISMATCH | ckpt: [5120,768] vs model:[6400,768]
```

## Root cause
Three loader bugs (all fixed) then one Tier B weight-format incompatibility:

**Loader bugs (fixed):**

1. `glm_4_7_flash_ggml_org_gguf` (imported first alphabetically) registers `deepseek2` in `GGUF_TO_FAST_CONVERTERS` but not `deepseek_v2`. The `load_gguf_checkpoint` patch remaps `model_type deepseek2→deepseek_v2` in the config dict returned to callers. The tokenizer init reads `gguf_param["config"]["model_type"]` = `"deepseek_v2"` and looks it up in `GGUF_TO_FAST_CONVERTERS` → `KeyError`. `glm_4_7_flash_gguf`'s guard (`if "deepseek2" in GGUF_SUPPORTED_ARCHITECTURES: return`) fires before registering `deepseek_v2` because ggml_org already ran first.

2. Many loaders (alphabetically after the GLM loaders) overwrite `modeling_utils.load_gguf_checkpoint` with narrow-sig `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrappers. When `from_pretrained` calls it with `model_to_load=dummy_model`, the narrow-sig wrapper raises `TypeError`.

3. `get_gguf_hf_weights_map` takes `model_type` from `hf_model.config.model_type` which is `"deepseek_v2"` (the HF config field). gguf-py's `MODEL_ARCH_NAMES` maps `MODEL_ARCH.DEEPSEEK2 → "deepseek2"`, not `"deepseek_v2"`.

**Tier B (unfixed):**

The deepseek2 GGUF format (ngxson/GLM-4.7-Flash-GGUF) uses a different weight layout than what `DeepseekV2ForCausalLM` expects:
- MoE expert weights stored as 3D stacked tensors `[hidden, inter, num_experts]` with separate gate/up; model expects combined `gate_up_proj [num_experts, 2*inter, hidden]` → all expert weights MISSING
- `kv_b_proj` stored as separate `attn_k_b [kv_lora_rank, qk_nope_head_dim, num_heads]` and `attn_v_b [kv_lora_rank, v_head_dim, num_heads]`; model expects combined `[kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim)]` → MISSING
- `q_b_proj` GGUF has `[num_heads*qk_nope_head_dim, q_lora_rank]` = `[5120, 768]`; model expects `[num_heads*(qk_nope_head_dim+qk_rope_head_dim), q_lora_rank]` = `[6400, 768]`; the rope query component is not stored separately in the GGUF (shared/derived)

## Fix
**Three loader fixes applied** (`tt_forge_models` branch `remediation/glm_4_7_flash_gguf-causal_lm-pytorch-4.7_Flash_GGUF-single_device-inference`):

**Fix 1** (`glm_4_7_flash_gguf/causal_lm/pytorch/loader.py` and `glm_4_7_flash_ggml_org_gguf/causal_lm/pytorch/loader.py`): Register `GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter` both in the early-return guard block (when another GLM loader ran first) and in the main registration body.

**Fix 2** (`glm_4_7_flash_gguf/causal_lm/pytorch/loader.py`): Added `_find_real_transformers_fn()` (BFS via `__globals__` + `__closure__`) to find the original transformers function through the narrow-sig wrapper chain. Added `_install_deepseek2_load_patch()` that uses this to install a wide-sig wrapper, called just-in-time in `_load_tokenizer()` and `load_model()` to survive clobbering.

**Fix 3** (`glm_4_7_flash_gguf/causal_lm/pytorch/loader.py`): Extended `_install_deepseek2_load_patch()` to also patch `gguf_utils.get_gguf_hf_weights_map`, remapping `deepseek_v2→deepseek2` before the gguf-py arch lookup.

**Tier B proposed fix** (not implemented): Implement a `DeepseekV2TensorProcessor` subclass registered as `TENSOR_PROCESSORS["deepseek_v2"]` that handles:
- Unstacking 3D MoE expert tensors and combining gate/up into `gate_up_proj`
- Combining separate `k_b`/`v_b` into `kv_b_proj`
- Padding `q_b_proj` with zeros for the rope component (or reconstructing from `k_pe`)

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

Implementing the deepseek2 GGUF tensor processor requires new infrastructure: a complete `TensorProcessor` subclass with custom logic for MoE expert tensor unstacking, MLA k/v projection merging, and q_b_proj shape reconstruction. This is not a scoped fix in one named function — it requires designing and implementing a new processing pipeline for a previously-unsupported GGUF dialect.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1610.38s (0:26:50) for final run reaching the Tier B failure
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/glm_4_7_flash_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/glm_4_7_flash_ggml_org_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5c585240a1c9e097d7c56b9179c6cd689fce1777 |
