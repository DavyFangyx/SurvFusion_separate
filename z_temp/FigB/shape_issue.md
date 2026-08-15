# Clinic Prompt Shape Issue

## Scope

This note is only about `Clinic` prompt features used by:

- `mlp_clinic_mean`
- `mlp_clinic_flatten`
- `snn_clinic_mean`
- `snn_clinic_flatten`
- `clinic_cox`

Relevant code:

- `utils/core_utils.py`
- `models/model_single_clinic.py`

## Current Behavior

The code currently infers only:

- `clinic_num_tokens`

That value is taken from the first dimension of a sample clinic `.pt` tensor:

- expected shape pattern: `[num_tokens, feature_dim]`
- inferred part: `num_tokens`

This means:

- if `L_i` and `D_i` differ only in token count, they are supported automatically
- example: `L_i = [6, 512]`, `D_i = [8, 512]` is fine

## Hard-Coded Assumption

For current clinic single-modal models, the per-token feature dimension is hard-coded as:

- `input_dim = 512`

The model does **not** dynamically infer the last dimension.
The model also does **not** automatically add a projection layer when the last dimension changes.

## Practical Conclusion

Case 1:

- `D_i` shape is `[N, 512]`
- no code change is needed

Case 2:

- `D_i` shape is `[N, C]` where `C != 512`
- current code will mismatch
- a code change is required

## What Must Be Changed If `C != 512`

At minimum, update the clinic model entry so `input_dim` is not fixed to `512`.

Current fixed points include:

- `utils/core_utils.py` model construction for `mlp_clinic_*`
- `utils/core_utils.py` model construction for `snn_clinic_*`
- `utils/core_utils.py` model construction for `clinic_cox`

Possible solutions:

1. Infer clinic feature dim from a sample `.pt` file and pass it into the model.
2. Keep model input fixed at `512`, but add an explicit projection layer from `C -> 512`.

## Recommended Rule For AI

Before changing code, check the actual clinic tensor shape:

- if shape is `[N, 512]`, do not modify model input logic
- if shape is `[N, C]` and `C != 512`, modify the clinic model input path

## Safe Summary

- token count is dynamic
- feature dim is currently fixed at `512`
- only token-count changes are already supported
- feature-dim changes are not supported yet
