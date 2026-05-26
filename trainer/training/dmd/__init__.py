# SPDX-License-Identifier: Apache-2.0
"""
DMD + self-forcing distillation training for HY-WorldPlay AR causal student.

Goal: distill HY-World1.5-Autoregressive-480P-I2V (50-step AR generator) into a
4-step student via Distribution Matching Distillation + self-forcing rollout.

Self-forcing algorithm follows LongLive (NVlabs/LongLive). KV cache uses
WorldPlay's existing per-chunk memory-frame prefill design unchanged
(no LongLive-style sink+local-window rolling, no LongLive prompt-switch
re-cache).

Reference checkpoint for sanity-checking distilled outputs:
HY-World1.5-Autoregressive-480P-I2V-distill (HF).
"""
