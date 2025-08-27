# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable


def _bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """统一的双线性插值（align_corners=False）"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DepthPriorExtractor(nn.Module):
    """
    深度先验提取（不可学习、无梯度）：
    - 梯度幅值（Sobel）
    - 局部方差（盒滤）
    - 有效/空洞掩码（基于深度范围）
    - （可选）RGB-深度边缘一致性
    """

    def __init__(self, var_kernel: int = 5, z_min: float = 0.0, z_max: float = 1.0,
                 use_rgb_edge: bool = True, robust_norm: bool = True) -> None:
        super().__init__()
        self.k = int(var_kernel)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_rgb_edge = bool(use_rgb_edge)
        self.robust_norm = bool(robust_norm)

        # Sobel 卷积核
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @torch.no_grad()
    def _robust_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        分位数归一化到 [0,1]，batch 内逐样本独立。
        若关闭 robust_norm，则退回到均值-方差或 min-max（这里用 min-max 更直观）。
        """
        B = x.shape[0]
        flat = x.view(B, -1)

        if self.robust_norm:
            p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True)
            p05 = torch.quantile(flat, 0.05, dim=1, keepdim=True)
            x = (flat - p05) / (p95 - p05 + 1e-6)
        else:
            # min-max 归一化（更快也更稳定）
            mn = flat.min(dim=1, keepdim=True).values
            mx = flat.max(dim=1, keepdim=True).values
            x = (flat - mn) / (mx - mn + 1e-6)

        return x.clamp_(0, 1).view_as(x)

    @torch.no_grad()
    def _compute_grad(self, x1: torch.Tensor) -> torch.Tensor:
        """Sobel 梯度幅值 + 归一化；输入 [B,1,H,W]"""
        # 边缘复制填充，避免边界衰减
        x_pad = F.pad(x1, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)
        g = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return self._robust_norm(g)

    @torch.no_grad()
    def _compute_var(self, x1: torch.Tensor) -> torch.Tensor:
        """局部方差 + 归一化；输入 [B,1,H,W]"""
        k = self.k
        pad = k // 2
        mu = F.avg_pool2d(x1, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        var = F.avg_pool2d(x1**2, kernel_size=k, stride=1, padding=pad, count_include_pad=False) - mu**2
        # 方差非负稳定
        var = var.clamp_min_(0.0)
        return self._robust_norm(var)

    @torch.no_grad()
    def _valid_and_hole(self, d1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """有效/空洞掩码；输入 [B,1,H,W]，深度范围由 z_min/z_max 控制"""
        valid = ((d1 > self.z_min) & (d1 < self.z_max)).float()
        hole = 1.0 - valid
        return valid, hole

    @torch.no_grad()
    def _edge_consistency(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        RGB-Depth 边缘一致性：[B,1,H,W]
        - rgb: [B,3,H,W] 或 [B,1,H,W]，值域任意（自动判断）
        - depth: [B,1,H,W]，值域已归一到 [0,1]
        """
        if rgb is None:
            return torch.ones_like(depth)

        if rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]

        # 归一化到 [0,1]
        if gray.max() > 1.0:
            gray = gray / 255.0

        g_rgb = self._compute_grad(gray)
        g_dep = self._compute_grad(depth)
        return (1.0 - (g_rgb - g_dep).abs()).clamp_(0, 1)

    @torch.no_grad()
    def forward(self, depth_raw: torch.Tensor, rgb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        输入:
          - depth_raw: [B,1,H,W]，mapper 已做裁剪/归一化到 [0,1]
          - rgb:       [B,3,H,W] 或 [B,1,H,W]，可选
        输出: dict，每个先验均为 [B,1,H,W]
        """
        d = depth_raw if depth_raw.dim() == 4 else depth_raw.unsqueeze(1)
        priors = {
            "gradient": self._compute_grad(d),
            "variance": self._compute_var(d),
        }
        valid, hole = self._valid_and_hole(d)
        priors["valid"] = valid
        priors["hole"] = hole

        if self.use_rgb_edge:
            priors["edge_consistency"] = self._edge_consistency(rgb, d)
        else:
            priors["edge_consistency"] = torch.ones_like(d)

        return priors


class ConfidencePredictor(nn.Module):
    """
    多尺度置信度预测：
    输入：投影后的 image/depth 特征 + 多通道先验
    输出：每尺度的 m \in (0,1)，并做温度控制与夹紧
    """

    def __init__(self, feature_dims: List[int], scale_keys: List[str],
                 hidden_dim: int = 128, temp_init: float = 1.5,
                 clamp_min: float = 0.05, clamp_max: float = 0.95) -> None:
        super().__init__()
        self.feature_dims = list(feature_dims)
        self.scale_keys = list(scale_keys)
        self.hidden = int(hidden_dim)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # 温度作为 buffer，避免 CPU/GPU 设备不一致
        self.register_buffer("temp", torch.tensor(float(temp_init), dtype=torch.float32))

        # image/depth 投影到同一隐层维度
        self.proj_image = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),  # 128//16=8 每组8通道，稳定
                nn.GELU(),
            ) for ch in self.feature_dims
        ])
        self.proj_depth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),
                nn.GELU(),
            ) for ch in self.feature_dims
        ])

        # 5 个先验通道（grad/var/valid/hole/edge）
        self.proj_prior = nn.Sequential(
            nn.Conv2d(5, self.hidden, 1, bias=False),
            nn.GroupNorm(16, self.hidden),
            nn.GELU(),
        )

        # 共享置信度 head
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden * 3, self.hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, self.hidden),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.hidden // 2),
            nn.GELU(),
            nn.Conv2d(self.hidden // 2, 1, 1)
        )

    def set_temperature(self, t: float) -> None:
        """更新温度（in-place），确保设备一致"""
        self.temp.fill_(float(t))

    def forward(self,
                image_features: Dict[str, torch.Tensor],
                depth_features: Dict[str, torch.Tensor],
                priors_ms: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        返回：每尺度的 m（置信度图），键与 scale_keys 对齐
        """
        m_maps: Dict[str, torch.Tensor] = {}

        for i, key in enumerate(self.scale_keys):
            if key not in image_features or key not in depth_features:
                # 缺失任何一侧特征则跳过（外侧做 RGB-only 回退）
                continue

            img = image_features[key]
            dep = depth_features[key]
            assert img.shape[2:] == dep.shape[2:], f"[MGM] {key} 空间尺寸不一致: {img.shape} vs {dep.shape}"

            # 投影
            img_p = self.proj_image[i](img)
            dep_p = self.proj_depth[i](dep)

            # 堆叠先验（固定 5 个通道）
            pri = priors_ms[key]
            prior_stack = torch.cat([
                pri["gradient"], pri["variance"], pri["valid"], pri["hole"], pri["edge_consistency"]
            ], dim=1)
            pri_p = self.proj_prior(prior_stack)

            x = torch.cat([img_p, dep_p, pri_p], dim=1)
            logits = self.head(x)

            # 温度控制 + 夹紧
            m = torch.sigmoid(logits / (self.temp + 1e-6))
            m_maps[key] = m.clamp(self.clamp_min, self.clamp_max)

        return m_maps


class MultiModalGatedFusion(nn.Module):
    """
    MGM 多模态门控融合（早期融合版本）：
    - 对每个尺度生成置信度 m
    - 采用 gated 残差融合: fused = m*Depth + (1-m)*Image + α*Image
    - 训练时提供三种正则/监督：
        * 置信度熵（鼓励 m 远离 0.5）
        * 空间方差（鼓励 m 非全常数）
        * 噪声掩码（可选，鼓励在噪点处 m 低）
    """

    @configurable
    def __init__(
        self,
        *,
        feature_dims: List[int],
        scale_keys: List[str],
        residual_alpha: float = 0.05,
        temp_init: float = 1.5,
        temp_final: float = 1.0,
        temp_steps: int = 5000,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
        loss_entropy_weight: float = 0.01,
        loss_var_weight: float = 0.01,
        noise_mask_weight: float = 0.1,
        hidden_dim: int = 128,
        # prior config
        prior_var_kernel: int = 5,
        prior_z_min: float = 0.0,
        prior_z_max: float = 1.0,
        prior_use_rgb_edge: bool = True,
        robust_norm: bool = True,
    ) -> None:
        super().__init__()

        self.feature_dims = list(feature_dims)
        self.scale_keys = list(scale_keys)
        self.residual_alpha = float(residual_alpha)

        # 温度调度
        self.temp_init = float(temp_init)
        self.temp_final = float(temp_final)
        self.temp_steps = int(temp_steps)
        self._cur_step = 0

        # Loss 系数
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.loss_var_weight = float(loss_var_weight)
        self.noise_mask_weight = float(noise_mask_weight)

        # 先验提取器（无梯度）
        self.prior = DepthPriorExtractor(
            var_kernel=prior_var_kernel,
            z_min=prior_z_min,
            z_max=prior_z_max,
            use_rgb_edge=prior_use_rgb_edge,
            robust_norm=robust_norm,
        )

        # 置信度预测器
        self.conf_pred = ConfidencePredictor(
            feature_dims=self.feature_dims,
            scale_keys=self.scale_keys,
            hidden_dim=hidden_dim,
            temp_init=self.temp_init,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

        # 对齐层（1x1+GN；不做激活，避免偏移统计量）
        self.align_image = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.GroupNorm(8, ch))
            for ch in self.feature_dims
        ])
        self.align_depth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.GroupNorm(8, ch))
            for ch in self.feature_dims
        ])

    @classmethod
    def from_config(cls, cfg):
        # 从 cfg 读取（没有就给出稳妥默认）
        mgm = cfg.MODEL.MGM
        prior = getattr(mgm, "PRIOR", None)

        feature_dims = list(mgm.get("FEATURE_DIMS", [96, 192, 384, 768]))
        scale_keys = list(mgm.get("SCALE_KEYS", ["res2", "res3", "res4", "res5"]))

        return dict(
            feature_dims=feature_dims,
            scale_keys=scale_keys,
            residual_alpha=mgm.RESIDUAL_ALPHA,
            temp_init=mgm.get("TEMP_INIT", 1.5),
            temp_final=mgm.get("TEMP_FINAL", 1.0),
            temp_steps=mgm.get("TEMP_STEPS", 5000),
            clamp_min=mgm.get("CLAMP_MIN", 0.05),
            clamp_max=mgm.get("CLAMP_MAX", 0.95),
            loss_entropy_weight=mgm.LOSS_ENTROPY_W,
            loss_var_weight=mgm.LOSS_VAR_W,
            noise_mask_weight=mgm.get("NOISE_MASK_WEIGHT", 0.1),
            hidden_dim=mgm.get("HIDDEN_DIM", 128),
            # prior 子配置
            prior_var_kernel=(prior.VAR_KERNEL if (prior and hasattr(prior, "VAR_KERNEL")) else 5),
            prior_z_min=(prior.Z_MIN if (prior and hasattr(prior, "Z_MIN")) else 0.0),
            prior_z_max=(prior.Z_MAX if (prior and hasattr(prior, "Z_MAX")) else 1.0),
            prior_use_rgb_edge=(prior.USE_RGB_EDGE if (prior and hasattr(prior, "USE_RGB_EDGE")) else True),
            robust_norm=mgm.get("ROBUST_NORM", True),
        )

    def _update_temperature(self) -> None:
        """线性温度调度：从 temp_init -> temp_final"""
        if self._cur_step <= self.temp_steps and self.temp_steps > 0:
            p = float(self._cur_step) / float(self.temp_steps)
            t = self.temp_init + (self.temp_final - self.temp_init) * p
            self.conf_pred.set_temperature(t)
        self._cur_step += 1

    def _extract_priors_ms(self,
                           depth_raw: torch.Tensor,
                           rgb_image: Optional[torch.Tensor],
                           target_sizes: Dict[str, Tuple[int, int]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        多尺度先验；所有插值使用 bilinear + align_corners=False
        """
        # 原分辨率先验（内部已 no_grad）
        pri_full = self.prior(depth_raw, rgb_image)

        # 多尺度下采样
        priors_ms: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, (h, w) in target_sizes.items():
            priors_ms[key] = {}
            for name, tensor in pri_full.items():
                priors_ms[key][name] = _bilinear(tensor, (h, w))
        return priors_ms

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        输入:
          - image_features: 多尺度 RGB 特征，键与 scale_keys 对齐
          - depth_features: 多尺度 Depth 特征，键与 scale_keys 对齐
          - depth_raw:      [B,1,H,W] 原始深度（已归一到 [0,1]）
          - rgb_image:      [B,3,H,W] 原图（可选，用于边缘一致性先验）
          - depth_noise_mask: [B,1,H,W] 可选噪声监督 (1=噪声/无效)
        输出:
          - fused_features: 融合后的多尺度特征
          - confidence_maps: 每尺度的 m
          - losses: MGM 相关 loss（训练时非空）
        """
        if is_training:
            self._update_temperature()

        # 目标尺寸来自 image_features（各尺度空间大小）
        target_sizes = {k: (v.shape[2], v.shape[3]) for k, v in image_features.items()}

        # 先验（多尺度）
        priors_ms = self._extract_priors_ms(depth_raw, rgb_image, target_sizes)

        # 预测置信度图
        m_maps = self.conf_pred(image_features, depth_features, priors_ms)

        # 融合
        fused: Dict[str, torch.Tensor] = {}
        eps = 1e-6

        for i, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue

            img = image_features[key]
            if key not in depth_features or key not in m_maps:
                # 回退：缺深度特征或缺 m 时，直接用对齐后的 RGB 特征
                fused[key] = self.align_image[i](img)
                continue

            dep = depth_features[key]
            assert img.shape[2:] == dep.shape[2:], f"[MGM] {key} 空间尺寸不一致: {img.shape} vs {dep.shape}"

            img_a = self.align_image[i](img)
            dep_a = self.align_depth[i](dep)
            m = m_maps[key]

            # 门控残差融合（注意系数 > 1 不一定是坏事，这里按设计保留 α*Image）
            fused[key] = m * dep_a + (1.0 - m) * img_a + self.residual_alpha * img_a

        # 训练期损失
        losses: Dict[str, torch.Tensor] = {}
        if is_training and len(m_maps) > 0:
            # 熵（越小越好）——鼓励 m 远离 0.5
            ent_all = []
            for m in m_maps.values():
                p = m.clamp(1e-4, 1 - 1e-4)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                ent_all.append(ent.mean())
            if ent_all:
                losses["loss_mgm_entropy"] = self.loss_entropy_weight * torch.stack(ent_all).mean()

            # 方差（越大越好）——避免全常数
            var_all = []
            for m in m_maps.values():
                var_all.append(m.var())
            if var_all:
                losses["loss_mgm_variance"] = self.loss_var_weight * (-torch.stack(var_all).mean())

            # 噪声掩码监督（1=噪声 => 期望 m 低；target_conf=1-noise）
            if depth_noise_mask is not None and depth_noise_mask.numel() > 0:
                noise_terms = []
                with torch.no_grad():
                    depth_noise_mask = depth_noise_mask.float()
                for m in m_maps.values():
                    H, W = m.shape[-2:]
                    tgt = 1.0 - _bilinear(depth_noise_mask, (H, W))
                    # 直接用概率空间 BCE（更简单；想更数值稳可改为 logits 形式并保存 logits）
                    noise_terms.append(F.binary_cross_entropy(m, tgt))
                if noise_terms:
                    losses["loss_mgm_noise"] = self.noise_mask_weight * torch.stack(noise_terms).mean()

        return fused, m_maps, losses


def build_mgm(cfg) -> MultiModalGatedFusion:
    """
    外部构建入口：支持 detectron2 的 @configurable
    用法：
      mgm = build_mgm(cfg)   # 等价于 MultiModalGatedFusion(cfg)
    """
    return MultiModalGatedFusion(cfg)
