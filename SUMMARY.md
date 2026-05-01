# Remediation Summary: mradermacher_qwen3_5_9b_antirep_v2_i1_gguf-causal_lm-pytorch-9B_Antirep_V2_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_5_9b_antirep_v2_i1_gguf/causal_lm/pytorch-9B_Antirep_V2_i1_GGUF-single_device-inference]

## Result
FAIL — qwen35 hybrid SSM+attention GGUF cannot be loaded as Qwen3: shape mismatch at full-attention layers

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-ssm-unsupported-arch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

Size mismatches at full-attention layers {3, 7, 11, 15, 19, 23, 27, 31}:
- `self_attn.q_proj.weight`: ckpt `[8192, 4096]` vs model `[2048, 4096]`
- `self_attn.k_proj.weight`: ckpt `[1024, 4096]` vs model `[512, 4096]`
- `self_attn.v_proj.weight`: ckpt `[1024, 4096]` vs model `[512, 4096]`
- `self_attn.o_proj.weight`: ckpt `[4096, 4096]` vs model `[4096, 2048]`
- `self_attn.q_norm.weight`: ckpt `[256]` vs model `[128]`
- `self_attn.k_norm.weight`: ckpt `[256]` vs model `[128]`

When the test runs in isolation, the immediate failure is an earlier OSError:
```
OSError: Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.
```
because the `qwen35`/`qwen3_5_text` tokenizer class is not registered in `GGUF_TO_FAST_CONVERTERS`. The RuntimeError is exposed when other tests in the same pytest session have already installed the global qwen35 patch (session contamination from tvall43/abliterated loaders).

## Root cause

The GGUF file is a genuine `qwen35` hybrid SSM+attention model (confirmed via metadata: `qwen35.ssm.conv_kernel`, `qwen35.ssm.state_size = [128]`, `qwen35.ssm.inner_size = [4096]`, `qwen35.full_attention_interval = [4]`). There are 32 GLA (GatedDeltaNet) layers and 8 full-attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31).

The current loader in `mradermacher_qwen3_5_9b_antirep_v2_i1_gguf/causal_lm/pytorch/loader.py` has no qwen35 architecture registration. When the qwen35→qwen3 alias is applied (as done by tvall43/abliterated-style loaders), `AutoModelForCausalLM` loads `Qwen3ForCausalLM` with uniform `num_attention_heads=16`, `head_dim=128`. The GGUF full-attention layers carry 64 Q-heads × 128 = 8192, but `Qwen3Config` creates q_proj of size `[2048, 4096]` (16 × 128). This is a fundamental shape mismatch: the qwen35→qwen3 alias is only valid for pure-attention Qwen3 models, not for Qwen3.5 hybrid SSM+attention models.

The correct model class is `Qwen3_5ForCausalLM` (available in transformers 5.2.0), but loading it from GGUF requires:
1. A `qwen35` entry in `GGUF_CONFIG_MAPPING` that maps SSM fields (`ssm.state_size`, `ssm.inner_size`, `ssm.conv_kernel`, `full_attention_interval`) to `Qwen3_5Config`
2. A `qwen3_5` → `qwen35` translation in `get_gguf_hf_weights_map`'s model_type normaliser (alongside the existing `qwen3_moe` → `qwen3moe` mapping)
3. New tensor name mappings in gguf-py's `QWEN35` arch for `Qwen3_5ForCausalLM` parameter names: `model.layers.N.linear_attn.in_proj_qkv.weight` → `blk.N.attn_qkv`, `model.layers.N.linear_attn.ssm_a.weight` → `blk.N.ssm_a`, etc. (gguf-py's current QWEN35 map uses `backbone.layers.N.mixer.*` naming, which does not match HF)

## Fix
Proposed fix (Tier B — new-infrastructure):

1. In `transformers/integrations/ggml.py`: add `"qwen35"` to `GGUF_CONFIG_MAPPING` with keys:
   ```python
   "qwen35": {
       "context_length": "max_position_embeddings",
       "block_count": "num_hidden_layers",
       "feed_forward_length": "intermediate_size",
       "embedding_length": "hidden_size",
       "rope.freq_base": "rope_theta",
       "attention.head_count": "num_attention_heads",
       "attention.head_count_kv": "num_key_value_heads",
       "attention.layer_norm_rms_epsilon": "rms_norm_eps",
       "attention.key_length": "head_dim",
       "full_attention_interval": "full_attention_interval",
       "vocab_size": "vocab_size",
   }
   ```
   Also add `GGUF_TO_FAST_CONVERTERS["qwen35"]` and `["qwen3_5_text"]` aliasing to `qwen3`'s converter.

2. In `transformers/modeling_gguf_pytorch_utils.py`, `get_gguf_hf_weights_map`: add
   ```python
   elif model_type in ("qwen3_5", "qwen3_5_text"):
       model_type = "qwen35"
   ```

3. In `gguf-py` (`gguf/tensor_mapping.py`): add mappings for `Qwen3_5ForCausalLM` parameter names to gguf QWEN35 tensor names:
   - `model.layers.N.linear_attn.in_proj_qkv` → `blk.N.attn_qkv`
   - `model.layers.N.linear_attn.out_proj` → `blk.N.attn_output`
   - `model.layers.N.linear_attn.gate` → `blk.N.attn_gate`
   - `model.layers.N.linear_attn.ssm_a` → `blk.N.ssm_a`
   - (and other SSM tensors)
   - `model.layers.N.self_attn.q_proj` → `blk.N.attn_q` (full-attention layers)
   - `model.layers.N.self_attn.k_proj` → `blk.N.attn_k`
   - `model.layers.N.self_attn.v_proj` → `blk.N.attn_v`

These changes span two repos (`transformers`, `gguf-py`) and require coordinated testing against multiple qwen35 models.

## Tier B justification
new-infrastructure — requires new tensor name mappings in gguf-py for `Qwen3_5ForCausalLM` parameter names (GLA SSM tensors have no existing mapping), plus new `GGUF_CONFIG_MAPPING` entries and model_type normaliser in transformers. Changes span two external repos and require coordinated validation.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
None — Tier B, no fix attempted.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
