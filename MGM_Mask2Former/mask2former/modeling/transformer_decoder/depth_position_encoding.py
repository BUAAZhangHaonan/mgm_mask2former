# -*- coding: utf-8 -*-
import torch
from torch import nn
from typing import Optional
from detectron2.config import configurable


class DepthPosEncoding(nn.Module):
    """
    深度图位置编码（可学习 + MLP）。
    1. 输入单通道深度 (B,1,H,W)（假设已标准化到[0,1]）
    2. 通过 1x1 MLP 提升维度
    3. 可学习缩放 alpha 控制整体幅值，避免主导二维位置编码
    4. mask=True 的位置视为无效/填充区，置零后再做 log1p 变换
    """

    @configurable
    def __init__(self, hidden_dim: int, beta: float = 10.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim 必须为偶数"
        # 动态 GN 组数：不超过32，且需整除
        G = min(32, hidden_dim)
        assert hidden_dim % G == 0, f"hidden_dim {hidden_dim} 不能被组数 {G} 整除"
        self.hidden_dim = hidden_dim
        self.beta = beta  # 可由配置注入
        self.mlp = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(G, hidden_dim),  # 动态组数
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(G, hidden_dim),
        )
        # 通道级缩放 (C,1,1)，更细粒度控制尺度
        self.alpha = nn.Parameter(torch.full((hidden_dim, 1, 1), 0.3))

    @classmethod
    def from_config(cls, cfg):
        # 兼容无 BETA 字段的情况
        beta = (
            getattr(cfg.MODEL.DPE, "BETA", 10.0) if hasattr(cfg.MODEL, "DPE") else 10.0
        )
        return {
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "beta": beta,
        }

    def forward(
        self, depth_raw: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            depth_raw: (B,1,H,W) 归一化深度
            mask: (B,H,W) 布尔张量, True 表示无效/填充区域(需要被置零)。
        返回:
            (B, hidden_dim, H, W) 深度位置编码
        """
        assert (
            depth_raw.dim() == 4 and depth_raw.shape[1] == 1
        ), f"期望输入 [B,1,H,W], 实际 {depth_raw.shape}"
        depth_float = depth_raw.float()

        if mask is not None:
            assert mask.dim() == 3, f"mask 需为 [B,H,W], 实际 {mask.shape}"
            # (~mask)=有效区域；无效区域清零，避免 log1p 幅值污染
            depth_float = depth_float * (~mask).unsqueeze(1)

        # 数值稳定：log1p(beta * d)
        depth_transformed = torch.log1p(self.beta * depth_float)

        depth_embedding = self.mlp(depth_transformed)
        return (depth_embedding * self.alpha).to(depth_raw.dtype)
