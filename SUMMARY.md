# Aura 7B GGUF Causal LM Fix Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[aura_7b_gguf/causal_lm/pytorch-7B_GGUF-single_device-inference]`

## Original Failure
```
2026-04-23 23:40:11.555 | critical | Always | TT_THROW: TIMEOUT: device timeout, potential hang detected
```
(locally reproduced as: `ValueError: accelerate is required when loading a GGUF file`)

## Root Cause
The `aura_7b_gguf/causal_lm/pytorch` loader had no `requirements.txt`. With `transformers==5.5.1`,
loading a GGUF checkpoint via `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` requires
two additional packages:
- `gguf>=0.10.0` — for parsing the GGUF binary format
- `accelerate` — required by transformers 5.5+ for GGUF model weight loading

Without these packages, the model load failed immediately with `ImportError` / `ValueError`,
which manifested as a device hang in the CI environment.

## Fix

### tt-forge-models (`remediation/aura-7b-gguf-fix`)
Added `aura_7b_gguf/causal_lm/pytorch/requirements.txt`:
```
gguf>=0.10.0
accelerate
```

Commit: `7f6f7950529e25003df69f5cb879ec79e3423d1f`

## Result
Test passes on n150 silicon with SILICON_PASS.
