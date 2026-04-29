# Remediation Summary: bartowski_thedrummer_valkyrie_49b_v2_1_gguf-causal_lm-pytorch-49B_V2_1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_thedrummer_valkyrie_49b_v2_1_gguf/causal_lm/pytorch-49B_V2_1_GGUF-single_device-inference]

## Result
XFAIL — 49B model Q4_K_M GGUF is 29 GB; n150 has 12 GB DRAM. Hardware capacity ceiling.

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-deci-arch-not-registered-in-autoconfig

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: Unrecognized model identifier: deci. Should contain one of afmoe, aimv2, ...

The GGUF file has `general.architecture = 'deci'` (DeciLM). Transformers 5.x has
'deci' in GGUF_CONFIG_MAPPING and GGUF_TO_FAST_CONVERTERS (uses GGUFLlamaConverter),
but it is NOT registered in AutoConfig's CONFIG_MAPPING. AutoTokenizer.from_pretrained
and AutoConfig.from_pretrained both call AutoConfig.for_model(model_type='deci') which
raises ValueError.

A second error surfaces after patching model_type to 'llama':
TypeError: unsupported operand type(s) for //: 'int' and 'list'
in LlamaConfig.__init__ at `self.head_dim = ... self.hidden_size // self.num_attention_heads`

This is because DeciLM uses per-layer lists for num_attention_heads, num_key_value_heads,
and intermediate_size (31 of 80 layers have head_count=0 — they are pure FFN layers
with no attention tensors at all).

## Root cause

Two loader-layer bugs combined with a hardware capacity ceiling:

1. **GGUF architecture 'deci' not registered in AutoConfig**: transformers 5.x has the
   field mapping for 'deci' in GGUF_CONFIG_MAPPING and the tokenizer converter
   (GGUFLlamaConverter) but does not register 'deci' in AutoConfig's CONFIG_MAPPING.
   So AutoConfig.for_model(model_type='deci') raises ValueError.

2. **Per-layer list fields in DeciLM GGUF**: The Valkyrie 49B v2.1 is a DeciLM
   architecture where 31 of 80 layers have zero attention heads (pure FFN layers).
   The GGUF stores num_attention_heads, num_key_value_heads, and intermediate_size
   as per-layer arrays (lists of 80 values), not scalars. LlamaConfig expects scalars.
   Unique values in GGUF: head_count ∈ {0, 64}, head_count_kv ∈ {0, 8},
   feed_forward_length ∈ {2816, 5632, 7168, 14336, 17920, 28672}.

3. **Hardware capacity**: TheDrummer_Valkyrie-49B-v2.1-Q4_K_M.gguf is 29 GB on disk.
   n150 has 12 GB DRAM. The model cannot fit on a single device regardless of any
   loader fix. Even loading just the quantized weights would require 2.4× the available
   DRAM.

A proper loader fix would require implementing a custom config and model class for the
DeciLM selective-attention + variable-FFN architecture. Standard LlamaForCausalLM
cannot represent this model correctly (missing attention tensors for 31 layers, and
different FFN shapes per layer).

## Fix

**Loader fix** (tt_forge_models remediation branch
`remediation/bartowski_thedrummer_valkyrie_49b_v2_1_gguf-causal_lm-pytorch-49B_V2_1_GGUF-single_device-inference`):

Added `_patch_deci_gguf_arch()` in
`bartowski_thedrummer_valkyrie_49b_v2_1_gguf/causal_lm/pytorch/loader.py` that:
- Patches `load_gguf_checkpoint` (and its module-level imports in tokenization_auto.py
  and configuration_utils.py) to remap `model_type = 'deci'` → `'llama'`
- Converts per-layer list fields (num_attention_heads, num_key_value_heads,
  intermediate_size) to scalars by taking the max non-zero value

This partial fix allows AutoConfig and AutoTokenizer to initialize without crashing.
It does not produce a correct model representation (LlamaForCausalLM cannot handle
selective attention or per-layer FFN variation), but it is sufficient to demonstrate
the hardware capacity ceiling.

**XFAIL config** (tt-xla remediation branch):
Added to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
```yaml
bartowski_thedrummer_valkyrie_49b_v2_1_gguf/causal_lm/pytorch-49B_V2_1_GGUF-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "49B model Q4_K_M GGUF is 29 GB; exceeds n150 12 GB DRAM. Hardware capacity ceiling."
```

## Verification
- pytest exit: not-run (XFAIL added before silicon run)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `bartowski_thedrummer_valkyrie_49b_v2_1_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1c8a01927b198e3faa8d52cecf2b243d557ec903 |
| tt-forge-models | d4e0f720838dce377efd0140d6b1ac22920c814b |
