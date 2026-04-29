# Remediation Summary: daniloreddy_qwen3_5_0_8b_gguf-causal_lm-pytorch-0.8B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch-0.8B_GGUF-single_device-inference]

## Result
FAIL — qwen35 is a hybrid Mamba2+Attention architecture; transformers has no model class for it and the qwen3 mapping produces catastrophic weight shape mismatches

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-mamba-hybrid-architecture-mismatch

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

Preceded by the following size-mismatch report for layers {3, 7, 11, 15, 19, 23}:
```
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.k_norm.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([256]) vs model:torch.Size([128])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.k_proj.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([512, 1024]) vs model:torch.Size([256, 1024])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.q_norm.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([256]) vs model:torch.Size([128])
```

Two earlier loader bugs were discovered and fixed (already present on branch
`arch-c-36-tt-xla-dev/nsmith/hf-bringup-45`) before the architecture
mismatch became visible:

1. `GGUF_FILE = "Q4_K_M.gguf"` → should be `"Qwen3.5-0.8B_Q4_K_M.gguf"` (the file
   does not exist at the old name on HuggingFace).
2. `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` narrow
   signature caused `TypeError: unexpected keyword argument 'model_to_load'`
   when transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=...)`.

Both are fixed on `hf-bringup-45` (commits `4c73c17225` and `6d42165a7f`).
The remediation branch is set to that tip.

## Root cause
The `daniloreddy/Qwen3.5-0.8B_GGUF` checkpoint (base model
`Qwen/Qwen3.5-0.8B-Base`) implements the **qwen35** GGUF architecture, which
is a **hybrid Mamba2+Attention model**, not a pure transformer.

GGUF metadata confirms the SSM components:
```
qwen35.ssm.conv_kernel: 4
qwen35.ssm.group_count: 16
qwen35.ssm.inner_size: 2048
qwen35.ssm.state_size: 128
qwen35.ssm.time_step_rank: 16
qwen35.full_attention_interval: 4
```

Inspecting the GGUF tensors:
- Blocks {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22} (18 blocks): hybrid
  Mamba2+Attention — have `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`,
  `ssm_out` tensors PLUS a fused `attn_qkv.weight [1024 × 6144]`.
- Blocks {3,7,11,15,19,23} (6 blocks, every `full_attention_interval=4`
  steps): pure full-attention — have separate `attn_q/k/v.weight` with
  `head_dim=256` and `num_key_value_heads=2`, giving
  `k_proj=[512,1024]` and `k_norm=[256]`.

The loader's `_patch_qwen35_support()` function maps the `qwen35` GGUF
architecture to the `qwen3` transformers model class (a pure transformer with
`head_dim=128`, `num_key_value_heads=2`, all-attention layers). This is
architecturally wrong:

1. The `qwen3` class has no Mamba2/SSM layers, so all 18 SSM block weights are
   silently ignored.
2. The full-attention blocks use `head_dim=256` in the GGUF, but the `qwen3`
   config derives `head_dim = hidden_size / num_heads = 1024 / 8 = 128`,
   producing `k_proj=[256,1024]` vs the GGUF's `[512,1024]`.  The 2× factor
   matches `head_dim_gguf / head_dim_qwen3 = 256 / 128 = 2`.
3. transformers raises `RuntimeError` because `ignore_mismatched_sizes=False`
   (the default).

## Fix
A correct fix requires implementing a `Qwen35ForCausalLM` transformers model
class that supports the hybrid Mamba2+Attention architecture:

- New model module analogous to `modeling_jamba.py` or `modeling_zamba2.py`
  in transformers, with:
  - SSM blocks using `group_count=16`, `inner_size=2048`, `state_size=128`
  - Full-attention blocks every `full_attention_interval=4` layers
  - Correct GQA: `num_q_heads=16`, `num_kv_heads=2`, `head_dim=256` for
    full-attention layers
- Registration of `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` pointing to
  the new model class (not `qwen3`).
- New `GGUF_TO_TRANSFORMERS_MAPPING` entries for qwen35 SSM fields.

This is upstream transformers work. Alternatively, a standalone loader could
implement the qwen35 architecture in PyTorch and load GGUF tensors by name,
bypassing `AutoModelForCausalLM.from_pretrained` entirely.

## Tier B justification
new-infrastructure: The qwen35 architecture is a hybrid Mamba2+Attention
model that has no corresponding transformers model class; a correct fix
requires implementing a new architecture (analogous to Jamba/Zamba2) across
multiple files in transformers and the model loader.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    88.89s (with hf-bringup-45 fixes; fails at model load, before silicon)
- Tier A attempts: N/A

## Files changed
No new files. The remediation branch in tt-forge-models points to
`arch-c-36-tt-xla-dev/nsmith/hf-bringup-45` tip (`6d42165a7f`), which
already contains the two prerequisite loader fixes:
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`:
  correct GGUF filename (`Qwen3.5-0.8B_Q4_K_M.gguf`)
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`:
  `_patched_load_gguf_checkpoint(*args, **kwargs)` wide signature

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | 2f7397a75f8aa674280280056bb936af5ddd1737   |
| tt-forge-models | 6d42165a7f1c9b48e0f115de198fe2a6be752c57   |
