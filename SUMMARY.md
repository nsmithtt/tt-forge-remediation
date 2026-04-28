# Remediation Summary: chexagent/pytorch-chexagent_2_3b-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[chexagent/pytorch-chexagent_2_3b-single_device-inference]

## Result
FAIL — pcc=0.49 (required ≥ 0.99); TT bfloat16 matmul accumulation is less precise than CPU bfloat16, causing error amplification across 56 transformer layers (24 SigLIP ViT + 32 Phi-2 LLM).

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
FAILED tests/runner/test_models.py::test_all_models_torch[chexagent/pytorch-chexagent_2_3b-single_device-inference]
AssertionError: pcc=0.49175 < required 0.99
```
Original failure (before loader fixes): `Fatal Python error: Segmentation fault`

## Root cause

Two layers of bugs were present:

### 1. Loader-layer bugs (all fixed — five commits in tt-forge-models)

CheXagent-2-3b uses transformers 4.40-era remote code that is incompatible with
transformers 5.x. Eleven issues were found and fixed across multiple sessions:

1. `is_tf_available` removed from `transformers.utils` — added `False`-returning stub.
2. `pad_token_id` missing from `CheXagentConfig.__init__` — patched config.
3. `rope_scaling` auto-populated with new `rope_type` dict by transformers 5.x — reset to None.
4. `CLIPModel.__init__` calls `AutoModel.from_pretrained` inside the meta-device context
   used by the outer `from_pretrained` — fixed by temporarily removing `DeviceContext`
   from the `TorchFunctionMode` stack.
5. `DeviceContext` restore in the `finally` block used wrong re-push order — fixed.
6. `apply_chat_template` returns a `BatchEncoding` in transformers 5.x, not a raw tensor —
   fixed by extracting `.input_ids`.
7. `_decode()` positional signature changed in transformers 5.x — fixed by passing
   `spaces_between_special_tokens` as a keyword argument.
8. `sample_image_url` pointed to a Wikipedia URL returning 403 — changed to an accessible
   HuggingFace datasets URL.
9. `SiglipVisionEmbeddings.position_ids` is a `persistent=False` buffer; transformers'
   `_move_missing_keys_from_meta_to_device` unconditionally replaces all non-persistent
   buffers with `torch.empty_like()` (garbage data), causing `IndexError` in the embedding
   lookup — fixed by re-creating `position_ids` in `load_model()` after `from_pretrained`.
10. `CLIPModel.forward` used `self.model(x, output_hidden_states=True).hidden_states[-1]`;
    in transformers 5.x `SiglipEncoder` no longer collects intermediate hidden states so
    `hidden_states` is always `None` — replaced with direct
    `self.model.encoder(inputs_embeds=self.model.embeddings(x)).last_hidden_state` call.
11. The inner `AutoModel.from_pretrained` for SigLIP loads weights as float32; the outer
    CheXagent `from_pretrained` uses `copy_()` for non-meta SigLIP tensors, preserving
    float32 — fixed by casting `model.model.visual` to checkpoint dtype after loading.
12. `PhiRotaryEmbedding` uses non-persistent `cos_cached`/`sin_cached` buffers that are
    zeroed out by `_move_missing_keys_from_meta_to_device` (same mechanism as issue 9),
    causing all-NaN attention outputs — fixed by re-calling `_set_cos_sin_cache` after
    `from_pretrained` in `load_model()`.
13. The image injection mechanism used `_parts`-based `torch.cat` that caused XLA view
    semantics errors — fixed by reworking the token assembly logic.

### 2. Silicon precision failure (unfixed — Tier B)

After all loader fixes, CPU inference passes. The silicon run compiles and executes
but the output logits have pcc=0.49 vs CPU (required: 0.99).

**Diagnostic measurements:**

| Scope | CPU fp32 vs CPU bf16 | CPU bf16 vs TT bf16 |
|---|---|---|
| Single SiglipEncoderLayer | — | pcc=0.99985 |
| LayerNorm only | — | pcc=1.0 |
| GELU only | — | pcc=0.9999997 |
| FC1 linear (1024→4096) | — | pcc=0.99997 |
| Self-attention (full) | — | pcc=0.99988 |
| 24-layer ViT encoder (CPU emb) | 0.99659 | 0.99091 |
| Full ViT encoder (TT emb+enc) | — | 0.98521 |
| Full visual encoder (+resampler) | — | 0.97781 |
| Full model (logits) | — | 0.49175 |

**Root cause:** TT hardware implements bfloat16 matmul with bfloat16 accumulation.
PyTorch on CPU executes bfloat16 matmul with float32 internal accumulation (the
`torch.mm` CPU kernel converts to fp32 before reducing). This divergence introduces
a small but real error per transformer layer (~0.012% noise/signal in the attention).
Over 24 SigLIP ViT layers the error compounds to pcc=0.991; the SigLIP resampler MLP
adds further error to pcc=0.977. The downstream Phi-2 LLM (32 layers, same precision
gap per layer) then amplifies the wrong visual features into uncorrelated logits
(pcc=0.49). The measured CPU bf16 vs CPU fp32 pcc=0.997 confirms bf16 itself causes
precision loss, but TT adds an additional 0.009 pcc gap on top.

**Disabling SDPA fusion** (`tt_enable_torch_fx_fusion_pass=False`) does not change
the pcc of the ViT encoder (still 0.985), ruling out the SDPAFusingPattern as the
source of error.

## Fix
The five loader-layer commits are in `tt-forge-models` on branch
`remediation/chexagent-pytorch-chexagent_2_3b-single_device-inference`.

**Proposed compiler-stack fix (not attempted — Tier B):**
Configure TT matmul operations to accumulate in float32 when inputs are bfloat16,
matching CPU semantics. This would require changes to the matmul lowering in
`tt-mlir` and/or the TT-metal matmul kernel launch in `tt-metal`. Because every
matmul in the model is affected, this is a cross-cutting change spanning many files
in at least two repos.

## Tier B justification
**cross-cutting**: Every bfloat16 matmul in the entire model is affected. Addressing
this requires changing the accumulation dtype policy for all matmul lowerings in
tt-mlir and tt-metal — it cannot be scoped to a single function or file.

## Verification
- pytest exit: FAIL (pcc=0.49 < required 0.99)
- Hardware: n150 (Wormhole; topology_discovery confirmed single chip)
- Duration: not recorded (test fails at pcc check)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/chexagent/pytorch/loader.py`
  — thirteen loader-layer fixes as described above (five commits)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4eccb183611ccf1568df2b392b1aca2bc8fd1ce4 |
| tt-forge-models | 6efa993201ad6e5e615b51152fec9a284b2a4499 |
