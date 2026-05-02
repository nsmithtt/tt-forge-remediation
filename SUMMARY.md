# Remediation Summary: mradermacher_qwen3_5_9b_claude_4_6_os_heretic_uncensored_instruct_gguf-causal_lm-pytorch-9B_CLAUDE_4_6_OS_HERETIC_UNCENSORED_INSTRUCT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_5_9b_claude_4_6_os_heretic_uncensored_instruct_gguf/causal_lm/pytorch-9B_CLAUDE_4_6_OS_HERETIC_UNCENSORED_INSTRUCT_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5-9B hybrid SSM+attention architecture has no transformers model class with per-layer head dispatch; loader fix applied but model load fails with size mismatch on full-attention layers

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-no-transformers-tensor-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix (qwen35 arch registered, wide-sig wrapper):
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

Size mismatches at full-attention layers {3, 7, 11, 15, 19, 23, 27, 31}:
- model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight: ckpt [8192, 4096] vs model [2048, 4096]
- model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight: ckpt [1024, 4096] vs model [512, 4096]
- model.layers.{3,7,11,15,19,23,27,31}.self_attn.v_proj.weight: ckpt [1024, 4096] vs model [512, 4096]
- model.layers.{3,7,11,15,19,23,27,31}.self_attn.o_proj.weight: ckpt [4096, 4096] vs model [4096, 2048]
- model.layers.{0...30}.self_attn.q_proj.weight: MISSING (loaded as GLA linear_attn, not self_attn)

## Root cause
Two bugs in sequence:

**Bug 1 (fixed):** The loader lacked `_patch_qwen35_support()` infrastructure. The GGUF file declares `architecture = qwen35` which is not in `GGUF_SUPPORTED_ARCHITECTURES`. Additionally, another model loader imported earlier during pytest collection had installed a narrow-sig `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` globally, which failed when transformers 5.x called it with `model_to_load=` kwarg.

**Bug 2 (Tier B):** Qwen3.5-9B is a hybrid architecture: 32 layers total, with full-attention at every 4th layer (indices 3, 7, 11, ..., 31) and GatedDeltaNet (GLA) linear-attention at the remaining 24 layers. Full-attention layers have 64 Q-heads / 16 KV-heads (q_proj=[8192,4096]) while GLA layers have 16 Q-heads / 4 KV-heads (q_proj=[2048,4096]). The transformers `Qwen3ForCausalLM` uses uniform head counts; there is no `Qwen3_5ForCausalLM` class with per-layer head dispatch, so loading the GGUF weights into the uniform model produces irrecoverable size mismatches on the 8 full-attention layers.

## Fix
**Bug 1 fix** (`tt_forge_models/mradermacher_qwen3_5_9b_claude_4_6_os_heretic_uncensored_instruct_gguf/causal_lm/pytorch/loader.py`):
- Added `_patch_qwen35_support()` function to register `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` and copy qwen3 mappings
- Added `_patched_load_gguf_checkpoint(*args, **kwargs)` with wide-sig to accept `model_to_load` kwarg from transformers 5.x
- Installed the patched function into all relevant transformers namespaces
- Added `enable_thinking=True` to `apply_chat_template`

**Bug 2 (proposed fix):** Requires a new `Qwen3_5ForCausalLM` model class in transformers that reads `full_attention_interval` from config and dispatches per-layer attention head counts (full-attn: `num_attention_heads=64, num_key_value_heads=16`; GLA: `num_attention_heads=16, num_key_value_heads=4`). Additionally requires a GGUF tensor name mapping for the hybrid architecture's GLA layer tensors (`blk.N.attn_qkv`, `blk.N.attn_gate`). This would live in `transformers/models/qwen3_5/` as a new model family.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

No transformers model class exists for the Qwen3.5 hybrid SSM+attention architecture; loading the per-layer-head GGUF weights requires a new `Qwen3_5ForCausalLM` class with per-layer head dispatch, which is new infrastructure touching multiple files.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    440.28s (7:20)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mradermacher_qwen3_5_9b_claude_4_6_os_heretic_uncensored_instruct_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c2803dbd4f98a5105922fe56c65133da87023eba |
| tt-forge-models | fb5e4cecb3d361c467ba837d9b2a1f7322e39d6a |
