"""Tuned RAB2 wrappers for focused rollout ablations.

These variants keep the same group-centered residual AB2 structure and compute
budget as the base RAB2 method, but reduce residual amplitude to trade a bit of
aggressiveness for better prompt-max tail behavior.
"""

from train_grpo_qwenimage_eff_mta_rab2 import _rab2_wrap


# Slightly conservative: keep most of base RAB2's mean gain while trimming
# large residual excursions.
sample_reference_model_rab2_mid = _rab2_wrap(0.4, 2.0, -1.0, 0.20, 1.25)

# More conservative: stronger shrink on rollout-specific residual magnitude.
sample_reference_model_rab2_tight = _rab2_wrap(0.4, 2.0, -1.0, 0.15, 1.20)

# Isolate "clip only": keep base bias but slightly tighten residual norm cap.
sample_reference_model_rab2_g130 = _rab2_wrap(0.4, 2.0, -1.0, 0.25, 1.30)

# Isolate "bias only": downweight extrapolated residual without changing cap.
sample_reference_model_rab2_b020 = _rab2_wrap(0.4, 2.0, -1.0, 0.20, 1.35)

# Slightly more aggressive than base: trust residual extrapolation a bit more.
sample_reference_model_rab2_b030 = _rab2_wrap(0.4, 2.0, -1.0, 0.30, 1.35)

# Continue exploring the "higher bias helps" direction near the current best.
sample_reference_model_rab2_b035 = _rab2_wrap(0.4, 2.0, -1.0, 0.35, 1.35)
sample_reference_model_rab2_b040 = _rab2_wrap(0.4, 2.0, -1.0, 0.40, 1.35)

# Keep the stronger residual trust but allow a slightly looser residual norm cap.
sample_reference_model_rab2_b030_g140 = _rab2_wrap(0.4, 2.0, -1.0, 0.30, 1.40)
