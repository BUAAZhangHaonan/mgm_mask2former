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

    def __init__(
        self,
        var_kernel: int = 5,
        z_min: float = 0.0,
        z_max: float = 1.0,
        use_rgb_edge: bool = True,
        robust_norm: bool = True,
        robust_norm_method: str = "minmax",
    ):
        super().__init__()
        self.k = int(var_kernel)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_rgb_edge = bool(use_rgb_edge)
        self.robust_norm = bool(robust_norm)
        self.robust_norm_method = robust_norm_method  # "quantile" or "minmax"

        # Sobel核
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @torch.no_grad()
    def _robust_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        归一化到[0,1]，逐样本独立；默认用 min-max（快且稳定）。
        若选择 'quantile'，会更鲁棒但明显更慢。
        """
        B = x.shape[0]
        flat = x.view(B, -1)
        if self.robust_norm and self.robust_norm_method == "quantile":
            # 注意：分位数很耗时；建议前期关闭或改用 minmax
            p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True)
            p05 = torch.quantile(flat, 0.05, dim=1, keepdim=True)
            x = (flat - p05) / (p95 - p05 + 1e-6)
        else:
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
        mu = F.avg_pool2d(
            x1, kernel_size=k, stride=1, padding=pad, count_include_pad=False
        )
        var = (
            F.avg_pool2d(
                x1**2, kernel_size=k, stride=1, padding=pad, count_include_pad=False
            )
            - mu**2
        )
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
    def forward(
        self, depth_raw: torch.Tensor, rgb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
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

    def __init__(
        self,
        feature_dims: List[int],
        scale_keys: List[str],
        hidden_dim: int = 128,
        temp_init: float = 1.5,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
        prior_in_channels: int = 5,
    ) -> None:
        super().__init__()
        self.feature_dims = list(feature_dims)
        self.scale_keys = list(scale_keys)
        self.hidden = int(hidden_dim)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # 温度作为 buffer，避免 CPU/GPU 设备不一致
        self.register_buffer(
            "temp", torch.tensor(float(temp_init), dtype=torch.float32)
        )

        # image/depth 投影到同一隐层维度
        self.proj_image = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    nn.GroupNorm(16, self.hidden),  # 128//16=8 每组8通道，稳定
                    nn.GELU(),
                )
                for ch in self.feature_dims
            ]
        )
        self.proj_depth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    nn.GroupNorm(16, self.hidden),
                    nn.GELU(),
                )
                for ch in self.feature_dims
            ]
        )

        # 最多 5 个先验通道（grad/var/valid/hole/edge）
        self.proj_prior = nn.Sequential(
            nn.Conv2d(prior_in_channels, self.hidden, 1, bias=False),
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
            nn.Conv2d(self.hidden // 2, 1, 1),
        )

    def set_temperature(self, t: float) -> None:
        """更新温度（in-place），确保设备一致"""
        self.temp.fill_(float(t))

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        priors_ms: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
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
            assert (
                img.shape[2:] == dep.shape[2:]
            ), f"[MGM] {key} 空间尺寸不一致: {img.shape} vs {dep.shape}"

            # 投影
            img_p = self.proj_image[i](img)
            dep_p = self.proj_depth[i](dep)

            # 堆叠先验（固定 5 个通道）
            pri = priors_ms[key]
            prior_stack = torch.cat(
                [
                    pri["gradient"],
                    pri["variance"],
                    pri["valid"],
                    pri["hole"],
                    pri["edge_consistency"],
                ],
                dim=1,
            )
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
        prior_enabled: bool = True,
        prior_use_grad: bool = True,
        prior_use_var: bool = True,
        prior_use_valid_hole: bool = True,
        prior_use_rgb_edge: bool = False,
        prior_var_kernel: int = 5,
        prior_z_min: float = 0.0,
        prior_z_max: float = 1.0,
        robust_norm: bool = False,
        robust_norm_method: str = "minmax",
        prior_compute_on: str = "full",  # "full"|"res2"|"res3"|"res4"|"res5"
        post_fuse_norm: bool = True
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

        self.prior_enabled = prior_enabled
        self.prior_use_grad = prior_use_grad
        self.prior_use_var = prior_use_var
        self.prior_use_valid_hole = prior_use_valid_hole
        self.prior_use_rgb_edge = prior_use_rgb_edge
        self.prior_compute_on = prior_compute_on
        self.post_fuse_norm = post_fuse_norm

        # 先验提取器
        self.prior = DepthPriorExtractor(
            var_kernel=prior_var_kernel,
            z_min=prior_z_min,
            z_max=prior_z_max,
            use_rgb_edge=prior_use_rgb_edge,
            robust_norm=robust_norm,
            robust_norm_method=robust_norm_method,
        )

        # 统计启用的先验通道数（梯度/方差/valid/hole/edge）
        prior_ch = 0
        if prior_enabled:
            if prior_use_grad:
                prior_ch += 1
            if prior_use_var:
                prior_ch += 1
            if prior_use_valid_hole:
                prior_ch += 2  # valid + hole
            if prior_use_rgb_edge:
                prior_ch += 1
        else:
            prior_ch = 0

        self.conf_pred = ConfidencePredictor(
            feature_dims=self.feature_dims,
            scale_keys=self.scale_keys,
            hidden_dim=hidden_dim,
            temp_init=temp_init,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            prior_in_channels=prior_ch,
        )

        # 对齐层
        self.align_image = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.GroupNorm(8, ch))
                for ch in self.feature_dims
            ]
        )
        self.align_depth = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.GroupNorm(8, ch))
                for ch in self.feature_dims
            ]
        )
        # 可选融合后归一化
        self.post_norm = (
            nn.ModuleList([nn.GroupNorm(8, ch) for ch in self.feature_dims])
            if self.post_fuse_norm
            else None
        )

    @classmethod
    def from_config(cls, cfg):
        mgm = cfg.MODEL.MGM
        prior = mgm.PRIOR

        feature_dims = list(mgm.get("FEATURE_DIMS", [96, 192, 384, 768]))
        scale_keys = list(mgm.get("SCALE_KEYS", ["res2", "res3", "res4", "res5"]))

        return dict(
            feature_dims=feature_dims,
            scale_keys=scale_keys,
            residual_alpha=mgm.RESIDUAL_ALPHA,
            temp_init=mgm.TEMP_INIT,
            temp_final=mgm.TEMP_FINAL,
            temp_steps=mgm.TEMP_STEPS,
            clamp_min=mgm.CLAMP_MIN,
            clamp_max=mgm.CLAMP_MAX,
            loss_entropy_weight=mgm.LOSS_ENTROPY_W,
            loss_var_weight=mgm.LOSS_VAR_W,
            noise_mask_weight=mgm.NOISE_MASK_WEIGHT,
            hidden_dim=mgm.HIDDEN_DIM,
            # prior
            prior_enabled=prior.ENABLED,
            prior_use_grad=prior.USE_GRADIENT,
            prior_use_var=prior.USE_VARIANCE,
            prior_use_valid_hole=prior.USE_VALID_HOLE,
            prior_use_rgb_edge=prior.USE_RGB_EDGE,
            prior_var_kernel=prior.VAR_KERNEL,
            prior_z_min=prior.Z_MIN,
            prior_z_max=prior.Z_MAX,
            robust_norm=mgm.ROBUST_NORM.ENABLED,
            robust_norm_method=mgm.ROBUST_NORM.METHOD,
            prior_compute_on=prior.COMPUTE_ON,
            post_fuse_norm=mgm.POST_FUSE_NORM,
        )

    def _update_temperature(self) -> None:
        """线性温度调度：从 temp_init -> temp_final"""
        if self._cur_step <= self.temp_steps and self.temp_steps > 0:
            p = float(self._cur_step) / float(self.temp_steps)
            t = self.temp_init + (self.temp_final - self.temp_init) * p
            self.conf_pred.set_temperature(t)
        self._cur_step += 1

    def _extract_priors_ms(
        self, depth_raw, rgb_image, target_sizes
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        多尺度先验：
        - PRIOR.COMPUTE_ON != "full" 时，先把 depth/rgb 下采样到某个尺度再算先验（提速明显），
          然后再插值到各尺度。
        - 全过程 no_grad，先验不参与反传。
        """
        if not self.prior_enabled:
            return {k: {} for k in target_sizes.keys()}

        # 选择先验计算分辨率
        if self.prior_compute_on == "full":
            d_for_prior = depth_raw
            rgb_for_prior = rgb_image
        else:
            assert (
                self.prior_compute_on in self.scale_keys
            ), "PRIOR.COMPUTE_ON 必须在 SCALE_KEYS 或 'full' 内"
            h0, w0 = next(
                (H, W)
                for k, (H, W) in target_sizes.items()
                if k == self.prior_compute_on
            )
            d_for_prior = _bilinear(depth_raw, (h0, w0))
            rgb_for_prior = (
                None if rgb_image is None else _bilinear(rgb_image, (h0, w0))
            )

        pri_full = self.prior(d_for_prior, rgb_for_prior)  # no_grad

        # 根据开关筛选并下采样到各尺度
        def pick(pr):
            arr = []
            if self.prior_use_grad:
                arr.append(pr["gradient"])
            if self.prior_use_var:
                arr.append(pr["variance"])
            if self.prior_use_valid_hole:
                arr += [pr["valid"], pr["hole"]]
            if self.prior_use_rgb_edge:
                arr.append(pr["edge_consistency"])
            if len(arr) == 0:
                # 没开任何先验：返回空dict（外面会跳过prior分支）
                return None
            return torch.cat(arr, dim=1)

        priors_ms = {}
        for key, (h, w) in target_sizes.items():
            sel = pick(pri_full)
            if sel is None:
                priors_ms[key] = {}
                continue
            priors_ms[key] = {"stack": _bilinear(sel, (h, w))}
        return priors_ms

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        输入:
        - image_features: 多尺度 RGB 特征，键与 scale_keys 对齐
        - depth_features: 多尺度 Depth 特征，键与 scale_keys 对齐
        - depth_raw: [B,1,H,W]，已对齐并归一到 [0,1]
        - rgb_image: [B,3,H,W]，可选（用于 RGB-D 边缘一致性）
        - depth_noise_mask: [B,1,H,W]，可选弱监督(1=噪声/无效)
        输出:
        - fused_features: 融合后的多尺度特征
        - confidence_maps: 每尺度的 m
        - losses: MGM 相关 loss（训练时非空）
        """

        # === 0) 训练期温度退火（确保真的在每步更新） ===
        if is_training:
            self._update_temperature()

        # === 1) 目标尺寸来自 image_features（各尺度空间大小） ===
        target_sizes = {k: (v.shape[2], v.shape[3]) for k, v in image_features.items()}

        # === 2) 先验多尺度计算 ===
        # 配置开关（均带默认，避免老配置报错）
        prior_enabled = getattr(self, "prior_enabled", True)
        prior_use_grad = getattr(self, "prior_use_grad", True)
        prior_use_var = getattr(self, "prior_use_var", True)
        prior_use_valid_hole = getattr(self, "prior_use_valid_hole", True)
        prior_use_rgb_edge = getattr(self, "prior_use_rgb_edge", False)

        if prior_enabled:
            # 内部已 no_grad；根据 compute_on / use_rgb_edge 等开关做实际计算
            priors_ms = self._extract_priors_ms(depth_raw, rgb_image, target_sizes)
            # 注意：_extract_priors_ms 可能返回全部通道，为省时与消融，这里根据开关把未启用的通道置零
            for key, dct in priors_ms.items():
                # 若 _extract_priors_ms 返回的是堆叠好的单键 "stack"，这里拆不出来；你的版本返回逐项键，按下方处理：
                if "gradient" in dct:
                    if not prior_use_grad:
                        dct["gradient"] = torch.zeros_like(dct["gradient"])
                    # 已启用 => 保留
                if "variance" in dct:
                    if not prior_use_var:
                        dct["variance"] = torch.zeros_like(dct["variance"])
                if "valid" in dct and "hole" in dct:
                    if not prior_use_valid_hole:
                        dct["valid"] = torch.zeros_like(dct["valid"])
                        dct["hole"] = torch.zeros_like(dct["hole"])
                if "edge_consistency" in dct:
                    if not prior_use_rgb_edge:
                        dct["edge_consistency"] = torch.zeros_like(dct["edge_consistency"])
        else:
            # 完全禁用先验：构造一个“全零先验”的空字典，供 Predictor 兼容（仍然会走5通道接口，但无需计算）
            priors_ms = {}
            for k, (H, W) in target_sizes.items():
                zeros = torch.zeros(
                    (depth_raw.shape[0], 1, H, W),
                    device=depth_raw.device,
                    dtype=depth_raw.dtype,
                )
                priors_ms[k] = {
                    "gradient": zeros,
                    "variance": zeros,
                    "valid": zeros,
                    "hole": zeros,
                    "edge_consistency": zeros,
                }

        # === 3) 预测置信度图 m（B×1×H_l×W_l） ===
        # 兼容现有 ConfidencePredictor.forward(image_features, depth_features, priors_ms)
        m_maps = self.conf_pred(image_features, depth_features, priors_ms)

        # === 4) 门控融合（含对齐层 + 可选融合后归一化） ===
        fused: Dict[str, torch.Tensor] = {}
        use_post_norm = (
            hasattr(self, "post_fuse_norm")
            and getattr(self, "post_fuse_norm")
            and hasattr(self, "post_norm")
            and self.post_norm is not None
        )

        for i, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue

            img = image_features[key]
            if key not in depth_features or key not in m_maps:
                # 回退：缺深度特征或缺 m 时，直接用对齐后的 RGB 特征
                out = self.align_image[i](img)
                if use_post_norm:
                    out = self.post_norm[i](out)
                fused[key] = out
                continue

            dep = depth_features[key]
            assert (
                img.shape[2:] == dep.shape[2:]
            ), f"[MGM] {key} 空间尺寸不一致: {img.shape} vs {dep.shape}"

            img_a = self.align_image[i](img)
            dep_a = self.align_depth[i](dep)
            m = m_maps[key]  # B×1×H×W，已clamp在[clamp_min, clamp_max]

            # 门控残差融合：m·Depth + (1-m)·Image + α·Image
            out = m * dep_a + (1.0 - m) * img_a + self.residual_alpha * img_a
            if use_post_norm:
                out = self.post_norm[i](out)
            fused[key] = out

        # === 5) 训练期损失（全部为“正损失”定义，数值稳定） ===
        losses: Dict[str, torch.Tensor] = {}
        if is_training and len(m_maps) > 0:
            # (a) 熵：鼓励远离0.5（越小越好）
            ent_terms = []
            for m in m_maps.values():
                p = m.clamp(1e-4, 1.0 - 1e-4)
                ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
                ent_terms.append(ent.mean())
            if ent_terms:
                losses["loss_mgm_entropy"] = (
                    self.loss_entropy_weight * torch.stack(ent_terms).mean()
                )

            # (b) 方差：希望 m 在空间上有分辨力（越大越好）=> 正损失用 (1 - var)
            var_terms = []
            for m in m_maps.values():
                var_terms.append(m.var(unbiased=False))
            if var_terms:
                mean_var = torch.stack(var_terms).mean()
                losses["loss_mgm_variance"] = self.loss_var_weight * (1.0 - mean_var)

            # (c) 噪声掩码弱监督（可选）：1=噪声 => 期望 m 低 ⇒ 目标 t=1-noise
            nmw = getattr(self, "noise_mask_weight", 0.0)
            if nmw > 0.0 and depth_noise_mask is not None and depth_noise_mask.numel() > 0:
                with torch.no_grad():
                    depth_noise_mask = depth_noise_mask.float()
                bces = []
                for m in m_maps.values():
                    H, W = m.shape[-2:]
                    tgt = 1.0 - _bilinear(depth_noise_mask, (H, W))
                    bces.append(F.binary_cross_entropy(m, tgt))
                losses["loss_mgm_noise"] = nmw * torch.stack(bces).mean()

        return fused, m_maps, losses


def build_mgm(cfg) -> MultiModalGatedFusion:
    """
    外部构建入口：支持 detectron2 的 @configurable
    用法：
      mgm = build_mgm(cfg)   # 等价于 MultiModalGatedFusion(cfg)
    """
    return MultiModalGatedFusion(cfg)
