# -*- coding: utf-8 -*-
"""
GSE-xLSTM: Group-Spatial-Enhanced xLSTM for Time Series Forecasting.

A dual-branch architecture combining Patch-based sLSTM temporal modelling
with Grouped Spatial MLP mixing, plus an optional NLinear baseline.

Architecture:
  Branch 1 -- Per-Variate Patched sLSTM
  Branch 2 -- Grouped Spatial MLP
  Fusion   -- Fixed-weight additive fusion with optional NLinear residual
  Norm     -- RevIN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)

from gse_xlstm.layers.StandardNorm import Normalize as RevIN
from .base_model import BaseModel


# ---------------------------------------------------------------------------
# CorrectedSpatialMLP -- Grouped spatial mixing block
# ---------------------------------------------------------------------------

class CorrectedSpatialMLP(nn.Module):
    """Grouped Spatial MLP with learnable soft-assignment matrix.

    Compresses the time axis first, then performs two-stage mixing on the
    spatial (node) dimension:  Group Mixing followed by Channel Mixing.

    Input:  [B, N, D]   (N = number of variates / nodes, D = hidden dim)
    Output: [B, N, D]
    """

    def __init__(self, num_nodes, hidden_dim, num_groups, dropout=0.1, hidden_mult=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim

        # Learnable soft-assignment matrix  G: [N, g]
        self.G = nn.Parameter(torch.randn(num_nodes, num_groups) * 0.02)

        # Group mixer: operates on the group dimension (g)
        self.group_mixer = nn.Sequential(
            nn.Linear(num_groups, num_groups * hidden_mult),
            nn.GELU(),
            nn.Linear(num_groups * hidden_mult, num_groups),
        )

        # Channel mixer: operates on the feature dimension (D)
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * hidden_mult, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        # Soft-assign nodes to groups (each node's weights sum to 1)
        G_norm = F.softmax(self.G, dim=1)  # [N, g]

        # Step 1: Grouping  -- [B, N, D] x [N, g] -> [B, g, D]
        x_grouped = torch.einsum('bnd, ng -> bgd', x, G_norm)

        # Step 2: Group Mixing (along group axis, with residual)
        x_permuted = x_grouped.permute(0, 2, 1)    # [B, D, g]
        x_mixed = self.group_mixer(x_permuted)      # [B, D, g]
        x_mixed = x_mixed.permute(0, 2, 1)          # [B, g, D]
        x_grouped = x_grouped + x_mixed

        # Step 3: Channel Mixing (along feature axis, with residual)
        x_channel = self.channel_mixer(x_grouped)
        x_grouped = x_grouped + x_channel

        # Step 4: Un-grouping  -- [B, g, D] x [N, g] -> [B, N, D]
        x_out = torch.einsum('bgd, ng -> bnd', x_grouped, G_norm)

        return self.norm(residual + x_out)


# ---------------------------------------------------------------------------
# GSEXlstm -- Main model
# ---------------------------------------------------------------------------

class GSEXlstm(BaseModel):
    """Group-Spatial-Enhanced xLSTM for multivariate time-series forecasting.

    The model consists of three components whose outputs are summed:

    1. **Patch sLSTM temporal branch** (Channel-Independent):
       Each variate is independently split into fixed-size patches, embedded,
       processed by a shared sLSTM stack, then projected to ``pred_len``.

    2. **Grouped Spatial MLP branch**:
       Time is compressed first, then spatial mixing is performed via
       ``CorrectedSpatialMLP`` (learnable group assignment + channel mixing).

    3. **NLinear baseline**:
       A simple per-variate linear mapping from ``seq_len`` to ``pred_len``
       that provides a stable residual prediction.

    The temporal and spatial branch outputs are combined with fixed weights
    (``time_branch_weight`` / ``spatial_branch_weight``), then added to the
    NLinear baseline when enabled.  RevIN normalises at input and denormalises
    at output.
    """

    def __init__(
        self,
        pred_len: int,
        seq_len: int,
        enc_in: int,
        xlstm_embedding_dim: int = 256,
        num_mem_tokens: int = 0,
        xlstm_dropout: float = 0.1,
        xlstm_conv1d_kernel_size: int = 0,
        xlstm_num_heads: int = 8,
        xlstm_num_blocks: int = 1,
        spatial_dim: int = 128,
        spatial_num_groups: int = 0,
        spatial_hidden_mult: int = 4,
        time_branch_weight: float = 1.0,
        spatial_branch_weight: float = 1.0,
        patch_size: int = 16,
        use_nlinear_baseline: bool = False,
    ) -> None:
        super().__init__(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)

        # --- Patch configuration ---
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        assert seq_len % patch_size == 0, (
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})"
        )

        # --- Branch weights ---
        self.time_branch_weight = time_branch_weight
        self.spatial_branch_weight = spatial_branch_weight

        # --- NLinear baseline (optional) ---
        self.use_nlinear_baseline = use_nlinear_baseline
        if use_nlinear_baseline:
            self.nlinear = nn.Linear(seq_len, pred_len)
        else:
            self.nlinear = None

        # --- Memory tokens for patched time branch ---
        self.num_time_mem_tokens = num_mem_tokens
        if num_mem_tokens > 0:
            self.time_mem_tokens = nn.Parameter(
                torch.randn(num_mem_tokens, xlstm_embedding_dim) * 0.01
            )
        else:
            self.time_mem_tokens = None

        # --- sLSTM configuration (shared) ---
        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=xlstm_num_heads,
                conv1d_kernel_size=xlstm_conv1d_kernel_size,
            )
        )

        # ==================================================================
        # Branch 1: Per-Variate Patched Time Branch
        # ==================================================================
        # Path: [B, T, N] -> patch -> embed -> sLSTM -> flatten -> project
        self.patch_embed = nn.Linear(patch_size, xlstm_embedding_dim)
        self.patch_xlstm = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=None,
                slstm_block=slstm_config,
                num_blocks=xlstm_num_blocks,
                embedding_dim=xlstm_embedding_dim,
                add_post_blocks_norm=True,
                dropout=xlstm_dropout,
                bias=True,
                slstm_at="all",
                context_length=self.num_patches + num_mem_tokens,
            )
        )
        # Project flattened (num_patches * D) -> pred_len per variate
        self.patch_head = nn.Linear(self.num_patches * xlstm_embedding_dim, pred_len)

        # ==================================================================
        # Branch 2: Spatial Branch
        # ==================================================================
        # Path: [B, T, N] -> [B, N, T] -> compress T->D -> spatial MLP -> predict
        self.spatial_dim = spatial_dim
        self.spatial_time_compress = nn.Linear(seq_len, self.spatial_dim)
        num_groups = spatial_num_groups if spatial_num_groups > 0 else max(4, enc_in // 8)
        self.spatial_mlp = CorrectedSpatialMLP(
            num_nodes=enc_in,
            hidden_dim=self.spatial_dim,
            num_groups=num_groups,
            dropout=xlstm_dropout,
            hidden_mult=spatial_hidden_mult,
        )
        self.spatial_pred_head = nn.Linear(self.spatial_dim, pred_len)

        # ==================================================================
        # RevIN (Reversible Instance Normalization)
        # ==================================================================
        self.reversible_instance_norm = RevIN(enc_in, affine=False)

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None):
        """Produce a multi-step forecast.

        Args:
            x_enc: Input tensor of shape ``[B, T, N]`` where *T* = ``seq_len``
                   and *N* = ``enc_in`` (number of variates).
            x_mark_enc: Unused -- kept for API compatibility.

        Returns:
            Prediction tensor of shape ``[B, pred_len, N]``.
        """
        # ---- RevIN: normalise ----
        x_enc = self.reversible_instance_norm(x_enc, "norm")
        B, T, N = x_enc.shape

       
        if self.use_nlinear_baseline and self.nlinear is not None:
            seq_last = x_enc[:, -1:, :].detach()        # [B, 1, N]
            x_base = x_enc - seq_last                    # centre on last value
            x_base = x_base.permute(0, 2, 1)             # [B, N, T]
            x_base = self.nlinear(x_base)                 # [B, N, pred_len]
            x_base = x_base.permute(0, 2, 1)             # [B, pred_len, N]
            x_base = x_base + seq_last                    # restore scale
        else:
            x_base = None

        
        # Branch 1: Per-Variate Patched Time Branch (sLSTM)
        
        # Step 1: Reshape into per-variate patches
        #   [B, T, N] -> [B, N, T] -> [B, N, num_patches, patch_size]
        x_patched = x_enc.permute(0, 2, 1)
        x_patched = rearrange(x_patched, 'b n (p ps) -> b n p ps', ps=self.patch_size)

        # Step 2: Embed each patch -> [B, N, num_patches, D]
        x_patched = self.patch_embed(x_patched)

        # Step 3: Flatten variate dimension for shared sLSTM
        #   [B, N, num_patches, D] -> [B*N, num_patches, D]
        x_patched = rearrange(x_patched, 'b n p d -> (b n) p d')

        # Step 4: Prepend memory tokens (shared across all variates)
        if self.time_mem_tokens is not None:
            m = repeat(self.time_mem_tokens, "m d -> bn m d", bn=B * N)
            x_patched, mem_ps = pack([m, x_patched], "bn * d")

        # Step 5: sLSTM processes each variate's patch sequence
        x_patched = self.patch_xlstm(x_patched)

        # Step 6: Remove memory tokens
        if self.time_mem_tokens is not None:
            _, x_patched = unpack(x_patched, mem_ps, "bn * d")

        # Step 7: Flatten patches -> project to pred_len
        x_patched = rearrange(x_patched, 'bn p d -> bn (p d)')
        x_time = self.patch_head(x_patched)                    # [B*N, pred_len]

        # Step 8: Restore shape -> [B, pred_len, N]
        x_time = rearrange(x_time, '(b n) p -> b p n', b=B, n=N)

        
        # Branch 2: Spatial Branch
        
        x_spatial = x_enc.permute(0, 2, 1)                    # [B, N, T]
        x_spatial = self.spatial_time_compress(x_spatial)      # [B, N, D]
        x_spatial = self.spatial_mlp(x_spatial)                # [B, N, D]
        x_spatial = self.spatial_pred_head(x_spatial)          # [B, N, pred_len]
        x_spatial = x_spatial.permute(0, 2, 1)                 # [B, pred_len, N]

        
        # Fusion
       
        x_fused = (self.time_branch_weight * x_time
                    + self.spatial_branch_weight * x_spatial)

        if self.use_nlinear_baseline and x_base is not None:
            x = x_base + x_fused
        else:
            x = x_fused

        # ---- RevIN: denormalise ----
        x = self.reversible_instance_norm(x, "denorm")
        return x
