# -*- coding: utf-8 -*-
import torch
from torch import nn
from typing import Optional

from detectron2.config import configurable


@configurable
class DepthPosEncoding(nn.Module):
    """
    Learned positional encoding for depth maps.
    Transforms a single-channel depth map into a high-dimensional embedding,
    with added stability and normalization.
    """

    def __init__(self, hidden_dim: int, beta: float = 10.0):
        super().__init__()
        assert (
            hidden_dim % 2 == 0
        ), "hidden_dim must be even for sinusoidal-like embeddings"
        self.hidden_dim = hidden_dim
        self.beta = beta  # A scaling factor for depth before log transform

        # A simple MLP with normalization to process the depth map
        self.mlp = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            # beta could be added to config if needed, e.g., cfg.MODEL.DPE.BETA
            "beta": 10.0,
        }

    def forward(
        self, depth_raw: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            depth_raw (Tensor): Raw depth map, normalized to [0, 1]. Shape: [B, 1, H, W]
            mask (Tensor, optional): Boolean padding mask. Shape: [B, H, W], True for padded areas.
        Returns:
            Tensor: The depth positional encoding. Shape: [B, hidden_dim, H, W]
        """
        assert (
            depth_raw.dim() == 4 and depth_raw.shape[1] == 1
        ), f"Expected depth_raw of shape [B, 1, H, W], got {depth_raw.shape}"

        # Ensure computation is in float32 for stability
        depth_float = depth_raw.float()

        # Mask out padded areas before any transformation
        if mask is not None:
            assert (
                mask.dim() == 3
            ), f"Expected mask of shape [B, H, W], got {mask.shape}"
            depth_float = depth_float.clone()
            depth_float[mask.unsqueeze(1)] = 0.0

        # Use log1p for better numerical stability with values in [0, 1]
        depth_transformed = torch.log1p(self.beta * depth_float)

        depth_embedding = self.mlp(depth_transformed)

        return depth_embedding.to(depth_raw.dtype)  # Cast back to original dtype
