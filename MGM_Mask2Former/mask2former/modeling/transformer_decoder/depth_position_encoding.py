# -*- coding: utf-8 -*-
import torch
from torch import nn

from detectron2.config import configurable


class DepthPosEncoding(nn.Module):
    """
    Learned positional encoding for depth maps.
    Takes a single-channel depth map and transforms it into a high-dimensional embedding.
    """

    @configurable
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim (int): The dimension of the output embedding, which should
                              match the transformer's hidden dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # A simple MLP implemented with 1x1 Convolutions to process the depth map
        # while preserving spatial dimensions.
        # 1 (log-depth) -> hidden_dim/2 -> hidden_dim
        self.mlp = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1, bias=False),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        }

    def forward(self, depth_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_raw (Tensor): Raw depth map, normalized to [0, 1].
                                Shape: [B, 1, H, W]
        Returns:
            Tensor: The depth positional encoding. Shape: [B, hidden_dim, H, W]
        """
        # Log-transform to better handle depth distribution
        # Add a small epsilon for stability
        depth_log = torch.log(depth_raw.clamp(min=1e-6))

        # Pass through the MLP
        depth_embedding = self.mlp(depth_log)

        return depth_embedding
