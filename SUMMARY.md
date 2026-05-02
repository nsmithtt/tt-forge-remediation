# Remediation Summary: qwen3_5_27b_claude_4_6_os_auto_variable_thinking_i1_gguf-causal_lm-pytorch-27B_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen3_5_27b_claude_4_6_os_auto_variable_thinking_i1_gguf/causal_lm/pytorch-27B_I1_GGUF-single_device-inference]

## Result
FAIL — qwen35 hybrid GDA+full-attention architecture has per-layer head count differences incompatible with a single Qwen3 transformers model class

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-hybrid-full-attn-head-count-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before fix):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing `_patched_load_gguf_checkpoint` to accept `**kwargs`:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

Mismatch report:
```
model.layers.{0...62}.self_attn.*                   | MISSING  |
model.layers.{3...63}.self_attn.q_proj.weight       | MISMATCH | ckpt: torch.Size([12288, 5120]) vs model: torch.Size([3072, 5120])
model.layers.{3...63}.self_attn.k_norm.weight       | MISMATCH | ckpt: torch.Size([256]) vs model: torch.Size([128])
model.layers.{3...63}.self_attn.q_norm.weight       | MISMATCH | ckpt: torch.Size([256]) vs model: torch.Size([128])
model.layers.{3...63}.self_attn.k_proj.weight       | MISMATCH | ckpt: torch.Size([1024, 5120]) vs model: torch.Size([512, 5120])
model.layers.{3...63}.self_attn.v_proj.weight       | MISMATCH | ckpt: torch.Size([1024, 5120]) vs model: torch.Size([512, 5120])
model.layers.{3...63}.self_attn.o_proj.weight       | MISMATCH | ckpt: torch.Size([5120, 6144]) vs model: torch.Size([5120, 3072])
```

## Root cause
The GGUF declares `general.architecture = qwen35` with `full_attention_interval = 4`. The GGUF metadata specifies GDA-layer config (`attention.head_count = 24`, `attention.head_count_kv = 4`, `attention.key_length = 256`), but the full-attention layers (indices 3, 7, 11, ..., 63) have a completely different head configuration: 96 Q-heads, 8 KV-heads, head_dim=128. The standard Qwen3 transformers model class has a single `num_attention_heads` and `num_key_value_heads` per model, so it creates all 64 layers with the GDA config (24 heads × 128 dim = 3072). When the GGUF full-attention tensors (12288 = 96 × 128) are loaded against this model, they mismatch.

Loader fix 1 (committed): 28 existing loaders had `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — a narrow signature that broke when transformers 5.2.0 added `model_to_load` as a keyword argument. The failing loader itself had no such patch at all. Fixed all 28 loaders to accept `**kwargs` and added the qwen35 patch (with correct signature) to the failing loader.

Loader fix 1 is insufficient: even after the kwargs fix, the qwen35 hybrid architecture is fundamentally incompatible with the Qwen3 model class.

## Fix
The fix would live in `transformers` (or a custom model class registered in `tt_forge_models`): implement a `Qwen3_5HybridModel` that:
1. Accepts per-layer attention type (GDA vs full-attention) based on `full_attention_interval`
2. Uses different `num_attention_heads`/`num_key_value_heads`/`head_dim` for full-attention vs GDA layers
3. Registers `qwen35` in `GGUF_CONFIG_MAPPING` and provides a proper `get_gguf_hf_weights_map` that maps the two layer types correctly

## Tier B justification
new-infrastructure: The qwen35 hybrid model requires a new model class in transformers supporting per-layer attention type dispatch and per-layer attention head configuration. No existing transformers class handles this.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    747.82s (0:12:27) for second run
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/qwen3_5_27b_claude_4_6_os_auto_variable_thinking_i1_gguf/causal_lm/pytorch/loader.py` — added qwen35 arch patch with `**kwargs`
- 28 other `tt_forge_models` loaders with `_patched_load_gguf_checkpoint` — added `**kwargs` to signature and pass-through

Remediation branch: `remediation/qwen3_5_27b_claude_4_6_os_auto_variable_thinking_i1_gguf-causal_lm-pytorch-27B_I1_GGUF-single_device-inference` in tt-forge-models

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0729432440baaf57a78b3b8b9a1e16d49d2deb36 |
