# SPDX-License-Identifier: Apache-2.0
"""
DMD-time transformer class.

The inference-side `HunyuanVideo_1_5_DiffusionTransformer` already supports the
full {AR forward with kv_cache, BI forward} forward methods needed by DMD:

    - generator  -> forward_txt(cache_txt=True) once, then per chunk:
                    forward_vision(cache_vision=True) prefill,
                    forward_vision(cache_vision=False) x 4 denoise steps
    - real_score -> forward_bi(...)                     bidirectional teacher
    - fake_score -> forward_bi(...)                     bidirectional critic

We sub-class it here only to attach the training-side `_fsdp_shard_conditions`
and gradient-checkpointing flag so that the same FSDP wrapping machinery used
by the rest of the trainer can pick it up. No source-level modification of the
inference transformer is performed.
"""

from hyvideo.models.transformers.worldplay_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)
from trainer.configs.models.dits.hunyuanvideo import HunyuanVideoConfig


class DMDHunyuanTransformer(HunyuanVideo_1_5_DiffusionTransformer):
    """Trainable variant of the WorldPlay HunyuanVideo transformer for DMD.

    Identical architecture & state-dict to the inference-side class. Adds
    only the trainer-side hooks needed for FSDP sharding and activation
    checkpointing.
    """

    _fsdp_shard_conditions = HunyuanVideoConfig()._fsdp_shard_conditions
    _supports_gradient_checkpointing = True
