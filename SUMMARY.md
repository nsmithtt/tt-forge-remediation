# Remediation Summary: gpt_oss_heretic_uncensored_gguf/causal_lm/pytorch-20B_INSTRUCT_Heretic_Uncensored_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_heretic_uncensored_gguf/causal_lm/pytorch-20B_INSTRUCT_Heretic_Uncensored_GGUF-single_device-inference]

## Result
FAIL — Fatal Python error: Segmentation fault in PJRT plugin when compiling 20B MoE model loaded from GGUF

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-segfault-gpt-oss-20b-moe-gguf

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Fatal Python error: Segmentation fault
[PJRT plugin crash during compilation of GptOssForCausalLM 20B MoE model]
```

In offline/isolated environments the loader itself also raises prior to hardware:

```
ValueError: GGUF model with architecture gpt-oss is not supported yet.
  File ".../transformers/configuration_utils.py", line 658, in _get_config_dict
      config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
  File ".../transformers/modeling_gguf_pytorch_utils.py", line 478, in load_gguf_checkpoint
      raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
```

## Root cause
Tier B PJRT-level crash. The GPT-OSS 20B MoE model loaded from a 15 GB Q4_K_M GGUF file causes a Fatal Python error (Segmentation fault) in the TT-XLA PJRT plugin during torch.compile("tt" backend). The model has a MoE architecture (gpt-oss.expert_count and gpt-oss.expert_used_count present in GGUF metadata) at 20B scale; this combination likely causes a crash in the compiler stack during graph lowering to StableHLO or during the tt-mlir compile stage. With HF Hub connectivity the GGUF dequantization proceeds (459 tensors, ~8 minutes), then segfaults. Without HF Hub connectivity the loader also fails because the gpt-oss GGUF architecture key is absent from GGUF_CONFIG_MAPPING in transformers 5.2.0 (though GptOssForCausalLM and GptOssConfig are present in transformers 5.2.0 as Python classes).

## Fix
No fix applied. Tier B compiler-stack crash is outside loader scope.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    >8 minutes before crash (GGUF dequantization: 459 tensors)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 1b343d762190004eb7a8d7be4598078370f720a3 |
