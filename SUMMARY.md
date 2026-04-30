# Remediation Summary: kimi_k2_5-pytorch-Kimi-K2.5-MXFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k2_5/pytorch-Kimi-K2.5-MXFP4-single_device-inference]

## Result
FAIL — moe_infer numpy dispatch; loader uses forbidden trimming + random weights; AMD Quark required for proper MXFP4 loading but not on standard PyPI

## Stack layer
loader

## Tier
B

## Bug fingerprint
moe-infer-numpy-dispatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}):
got AttributeError("'ndarray' object has no attribute 'add'")

from user code:
   File "modeling_deepseek.py", line 1448, in torch_dynamo_resume_in_forward_at_1429
    layer_outputs = decoder_layer(...)
   File "modeling_deepseek.py", line 1201, in forward
    hidden_states = self.mlp(hidden_states)
   File "modeling_deepseek.py", line 537, in forward
    y = self.moe_infer(hidden_states, topk_idx, topk_weight)
   File "modeling_deepseek.py", line 583, in moe_infer
    end_idx = start_idx + num_tokens

## Root cause
Two compounding problems:

**Primary (Tier B blocker): AMD Quark unavailable.** The `amd/Kimi-K2.5-MXFP4` checkpoint uses `quant_method: "quark"` with `weight_format: "real_quantized"` (MX-FP4 per-group weights). Loading these weights requires AMD Quark (`quark` package), which is AMD's proprietary quantization library distributed from AMD's own channels — not available on standard PyPI. Without Quark, the FP4-packed weights cannot be dequantized, so no real model weights can be loaded.

**Secondary (proximate error): moe_infer numpy dispatch.** The `DeepseekV3MoE.moe_infer` method in the model's remote code calls `tokens_per_expert.cpu().numpy()`, then iterates over the resulting numpy array. The per-token arithmetic `end_idx = start_idx + num_tokens` (where `num_tokens` is a numpy integer) fails in torch.compile's FakeTensor tracing because Dynamo tries to dispatch `ndarray.__add__` instead of `int.__add__`. This is the same failure class seen in DeepSeek V3, Kimi K2, and HunyuanDiT — the fix is to replace `moe_infer` with a static per-expert masked matmul that avoids the numpy-based dynamic routing.

**Compound effect of loader workarounds.** The current loader contains forbidden workarounds: it forces `text_config.num_hidden_layers = 2`, `text_config.hidden_size = 1024`, `text_config.num_attention_heads = 16`, `text_config.intermediate_size = 4096`, and constructs the model with random weights via `model_class(text_config)` (no checkpoint loaded). These make the proximate moe_infer failure appear in a toy model rather than the real architecture.

**Model size context.** The full text backbone has `num_hidden_layers=61`, `hidden_size=7168`, `n_routed_experts=384`. At MXFP4 (4-bit), the text backbone is approximately 12 GB — within n150's 12 GB DRAM limit. At BF16 (the only fallback without Quark), it is approximately 49 GB — exceeding all available single-device DRAM.

## Fix
The proper fix requires two parts in the loader (`kimi_k2_5/pytorch/loader.py` in tt-forge-models):

1. **AMD Quark integration**: Add AMD Quark as a requirement and load the full model via `AutoModelForCausalLM.from_pretrained("amd/Kimi-K2.5-MXFP4", trust_remote_code=True)`. This loads the real FP4 weights and constructs the actual architecture (61-layer DeepSeek V3 text backbone at ~12 GB). AMD Quark must be installed from `https://quark.docs.amd.com` — it is not installable via `pip install quark`.

2. **moe_infer static patch**: After loading, monkey-patch `DeepseekV3MoE.moe_infer` to replace the numpy-based token routing with a static per-expert masked matmul (same approach as used in deepseek_v3, kimi_k2, and hunyuan3d loaders). The patched version iterates over experts with boolean masks rather than slicing a numpy-indexed sorted token buffer.

## Tier B justification
new-infrastructure — AMD Quark is AMD's proprietary quantization runtime required to deserialize MX-FP4 tensors; it is not available on standard PyPI and cannot be installed as a normal dependency, making proper model loading impossible without out-of-band AMD toolchain installation.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    75.00s (1:15)
- Tier A attempts: N/A

## Files changed
None — no fixes applied.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
