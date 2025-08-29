# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

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
        在 AMP 下：
        - quantile 统计对半精度不稳定 -> 统一转 float32 计算后再 cast 回原 dtype
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

        if self.robust_norm and self.robust_norm_method == "quantile":
            x_flat_f32 = x_flat.float()
            p95 = torch.quantile(x_flat_f32, 0.95, dim=1, keepdim=True)
            p05 = torch.quantile(x_flat_f32, 0.05, dim=1, keepdim=True)
            norm_flat = (x_flat_f32 - p05) / (p95 - p05 + 1e-6)
            norm_flat = norm_flat.clamp_(0, 1).to(x.dtype)
        else:
            mn = x_flat.min(dim=1, keepdim=True).values
            mx = x_flat.max(dim=1, keepdim=True).values
            norm_flat = (x_flat - mn) / (mx - mn + 1e-6)
            norm_flat = norm_flat.clamp_(0, 1)

        return norm_flat.view_as(x)

    @torch.no_grad()
    def _compute_grad(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        w_x = self.sobel_x.to(x_pad.dtype)
        w_y = self.sobel_y.to(x_pad.dtype)
        gx = F.conv2d(x_pad, w_x, stride=1, padding=0)
        gy = F.conv2d(x_pad, w_y, stride=1, padding=0)
        g = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return self._robust_norm(g)

    @torch.no_grad()
    def _compute_var(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k
        pad = k // 2
        mu = F.avg_pool2d(
            x, kernel_size=k, stride=1, padding=pad, count_include_pad=False
        )
        var = (
            F.avg_pool2d(
                x**2, kernel_size=k, stride=1, padding=pad, count_include_pad=False
            )
            - mu**2
        )
        var = var.clamp_min_(0.0)
        return self._robust_norm(var)

    @torch.no_grad()
    def _valid_and_hole(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Using strict inequality assumes invalid/saturated depths are exactly 0.0 or 1.0.
        # This is generally a safe and robust assumption for normalized depth maps.
        valid = ((d > self.z_min) & (d < self.z_max)).float()
        hole = 1.0 - valid
        return valid, hole

    @torch.no_grad()
    def _edge_consistency(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # If the switch is off, return immediately.
        if not self.use_rgb_edge or rgb is None:
            return torch.ones_like(depth)

        if rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]

        if gray.max() > 1.0:
            gray = gray / 255.0

        g_rgb = self._compute_grad(gray)
        g_dep = self._compute_grad(depth)
        return (1.0 - (g_rgb - g_dep).abs()).clamp_(0, 1)

    @torch.no_grad()
    def forward(
        self, depth_raw: torch.Tensor, rgb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        d = depth_raw if depth_raw.dim() == 4 else depth_raw.unsqueeze(1)

        priors = {"gradient": self._compute_grad(d), "variance": self._compute_var(d)}
        valid, hole = self._valid_and_hole(d)
        priors["valid"] = valid
        priors["hole"] = hole
        priors["edge_consistency"] = self._edge_consistency(rgb, d)

        return priors


class ConfidencePredictor(nn.Module):
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

        assert len(feature_dims) == len(
            scale_keys
        ), "Feature dimensions and scale keys must match."
        assert hidden_dim % 16 == 0, "hidden_dim must be divisible by 16"

        self.register_buffer(
            "temp", torch.tensor(float(temp_init), dtype=torch.float32)
        )

        self.proj_image = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    nn.GroupNorm(16, self.hidden),
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

        self.proj_prior = None
        if prior_in_channels > 0:
            self.proj_prior = nn.Sequential(
                nn.Conv2d(prior_in_channels, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),
                nn.GELU(),
            )
        self._prior_channels = prior_in_channels

        in_ch_head = self.hidden * 3 if self._prior_channels > 0 else self.hidden * 2
        self.head = nn.Sequential(
            nn.Conv2d(in_ch_head, self.hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, self.hidden),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.hidden // 2),
            nn.GELU(),
            nn.Conv2d(self.hidden // 2, 1, 1),
        )

    def set_temperature(self, t: float) -> None:
        self.temp.fill_(float(t))

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        priors_ms: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        m_maps: Dict[str, torch.Tensor] = {}

        for i, key in enumerate(self.scale_keys):
            if key not in image_features or key not in depth_features:
                continue

            img_p = self.proj_image[i](image_features[key])
            dep_p = self.proj_depth[i](depth_features[key])
            features_to_cat = [img_p, dep_p]

            prior_stack = priors_ms.get(key, {}).get("stack", None)

            if prior_stack is not None and self.proj_prior is not None:
                if prior_stack.shape[1] != self._prior_channels:
                    raise RuntimeError(
                        f"Runtime prior channels {prior_stack.shape[1]} != initialized {self._prior_channels}"
                    )
                features_to_cat.append(self.proj_prior(prior_stack))

            x = torch.cat(features_to_cat, dim=1)
            logits = self.head(x)
            m = torch.sigmoid(logits / self.temp.clamp(1e-6))
            m_maps[key] = m.clamp(self.clamp_min, self.clamp_max)

        return m_maps


class MultiModalGatedFusion(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        feature_dims: List[int],
        scale_keys: List[str],
        residual_alpha: float,
        temp_init: float,
        temp_final: float,
        temp_steps: int,
        clamp_min: float,
        clamp_max: float,
        loss_entropy_weight: float,
        noise_mask_weight: float,
        hidden_dim: int,
        prior_enabled: bool,
        prior_use_grad: bool,
        prior_use_var: bool,
        prior_use_valid_hole: bool,
        prior_use_rgb_edge: bool,
        prior_var_kernel: int,
        prior_z_min: float,
        prior_z_max: float,
        robust_norm: bool,
        robust_norm_method: str,
        prior_compute_on: str,
        post_fuse_norm: bool,
    ) -> None:
        super().__init__()

        if prior_enabled and prior_compute_on != "full":
            assert (
                prior_compute_on in scale_keys
            ), f"`prior_compute_on` ('{prior_compute_on}') must be 'full' or one of `scale_keys` ({scale_keys})"

        self.feature_dims = list(feature_dims)
        self.scale_keys = list(scale_keys)
        self.residual_alpha = float(residual_alpha)
        self.temp_init = float(temp_init)
        self.temp_final = float(temp_final)
        self.temp_steps = int(temp_steps)
        self._cur_step = 0
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.noise_mask_weight = float(noise_mask_weight)
        self.prior_enabled = prior_enabled
        self.prior_use_grad = prior_use_grad
        self.prior_use_var = prior_use_var
        self.prior_use_valid_hole = prior_use_valid_hole
        self.prior_use_rgb_edge = prior_use_rgb_edge
        self.prior_compute_on = prior_compute_on
        self.post_fuse_norm = post_fuse_norm

        self.prior_extractor = DepthPriorExtractor(
            var_kernel=prior_var_kernel,
            z_min=prior_z_min,
            z_max=prior_z_max,
            use_rgb_edge=self.prior_use_rgb_edge,
            robust_norm=robust_norm,
            robust_norm_method=robust_norm_method,
        )

        prior_ch = 0
        if self.prior_enabled:
            if self.prior_use_grad:
                prior_ch += 1
            if self.prior_use_var:
                prior_ch += 1
            if self.prior_use_valid_hole:
                prior_ch += 2
            if self.prior_use_rgb_edge:
                prior_ch += 1

        self.conf_pred = ConfidencePredictor(
            feature_dims=self.feature_dims,
            scale_keys=self.scale_keys,
            hidden_dim=hidden_dim,
            temp_init=temp_init,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            prior_in_channels=prior_ch,
        )

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

        self.post_norm = (
            nn.ModuleList([nn.GroupNorm(8, ch) for ch in self.feature_dims])
            if self.post_fuse_norm
            else None
        )

        self._prior_missing_warned = False

    @classmethod
    def from_config(cls, cfg):
        mgm_cfg = cfg.MODEL.MGM
        prior_cfg = mgm_cfg.PRIOR
        return {
            "feature_dims": mgm_cfg.FEATURE_DIMS,
            "scale_keys": mgm_cfg.SCALE_KEYS,
            "residual_alpha": mgm_cfg.RESIDUAL_ALPHA,
            "temp_init": mgm_cfg.TEMP_INIT,
            "temp_final": mgm_cfg.TEMP_FINAL,
            "temp_steps": mgm_cfg.TEMP_STEPS,
            "clamp_min": mgm_cfg.CLAMP_MIN,
            "clamp_max": mgm_cfg.CLAMP_MAX,
            "loss_entropy_weight": mgm_cfg.LOSS_ENTROPY_W,
            "noise_mask_weight": mgm_cfg.NOISE_MASK_WEIGHT,
            "hidden_dim": mgm_cfg.HIDDEN_DIM,
            "prior_enabled": prior_cfg.ENABLED,
            "prior_use_grad": prior_cfg.USE_GRADIENT,
            "prior_use_var": prior_cfg.USE_VARIANCE,
            "prior_use_valid_hole": prior_cfg.USE_VALID_HOLE,
            "prior_use_rgb_edge": prior_cfg.USE_RGB_EDGE,
            "prior_var_kernel": prior_cfg.VAR_KERNEL,
            "prior_z_min": prior_cfg.Z_MIN,
            "prior_z_max": prior_cfg.Z_MAX,
            "robust_norm": mgm_cfg.ROBUST_NORM.ENABLED,
            "robust_norm_method": mgm_cfg.ROBUST_NORM.METHOD,
            "prior_compute_on": prior_cfg.COMPUTE_ON,
            "post_fuse_norm": mgm_cfg.POST_FUSE_NORM,
        }

    def _update_temperature(self) -> None:
        """
        在每个训练步骤中更新状态并应用温度调度。
        """
        # 行为守卫：只在训练模式下执行任何操作
        if self.training:
            # 逻辑判断：仅在退火阶段根据当前步数计算和设置温度
            if self._cur_step <= self.temp_steps and self.temp_steps > 0:
                p = float(self._cur_step) / float(self.temp_steps)
                t = self.temp_init + (self.temp_final - self.temp_init) * p
                self.conf_pred.set_temperature(t)

            # 状态更新：无论是否在退火阶段，只要是训练，步数计数器就必须增加
            self._cur_step += 1

    def _prepare_priors_ms(
        self,
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor],
        target_sizes: Dict[str, Tuple[int, int]],
        override_compute_on: Optional[str] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        if not self.prior_enabled:
            return {}

        # 若 forward 已经提供 override，则优先；否则沿用 self.prior_compute_on
        base_compute_on = (
            self.prior_compute_on
            if override_compute_on is None
            else override_compute_on
        )

        # 再次兜底（双保险）
        effective_compute_on = base_compute_on
        if effective_compute_on != "full" and effective_compute_on not in target_sizes:
            effective_compute_on = "full"

        if effective_compute_on == "full":
            compute_res_rgb = rgb_image
            compute_res_depth = depth_raw
        else:
            h0, w0 = target_sizes[effective_compute_on]
            compute_res_depth = _bilinear(depth_raw, (h0, w0))
            compute_res_rgb = (
                _bilinear(rgb_image, (h0, w0)) if rgb_image is not None else None
            )

        all_priors_single_res = self.prior_extractor(compute_res_depth, compute_res_rgb)

        priors_ms = {key: {} for key in target_sizes}
        for prior_name, prior_tensor in all_priors_single_res.items():
            for key, (h, w) in target_sizes.items():
                priors_ms[key][prior_name] = _bilinear(prior_tensor, (h, w))

        return priors_ms

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:

        if self.training:
            self._update_temperature()

        target_sizes = {k: v.shape[-2:] for k, v in image_features.items()}

        # 如果当前 batch 缺失 prior_compute_on 对应 level，回退到 full
        effective_prior_compute_on = self.prior_compute_on
        if (
            self.prior_enabled
            and self.prior_compute_on != "full"
            and self.prior_compute_on not in target_sizes
        ):
            if not self._prior_missing_warned:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[MGM] prior_compute_on='{self.prior_compute_on}' not present in this batch features. Fallback to 'full'."
                )
            self._prior_missing_warned = True
            effective_prior_compute_on = "full"

        priors_ms = self._prepare_priors_ms(
            depth_raw,
            rgb_image,
            target_sizes,
            override_compute_on=effective_prior_compute_on,
        )

        priors_for_pred = {}
        if self.prior_enabled:
            for key in target_sizes:
                prior_list = []
                if self.prior_use_grad:
                    prior_list.append(priors_ms[key]["gradient"])
                if self.prior_use_var:
                    prior_list.append(priors_ms[key]["variance"])
                if self.prior_use_valid_hole:
                    prior_list.append(priors_ms[key]["valid"])
                    prior_list.append(priors_ms[key]["hole"])
                if self.prior_use_rgb_edge:
                    prior_list.append(priors_ms[key]["edge_consistency"])

                if prior_list:
                    priors_for_pred[key] = {"stack": torch.cat(prior_list, dim=1)}

        m_maps = self.conf_pred(image_features, depth_features, priors_for_pred)

        fused: Dict[str, torch.Tensor] = {}
        for i, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue

            img_a = self.align_image[i](image_features[key])

            if key not in depth_features or key not in m_maps:
                fused[key] = self.post_norm[i](img_a) if self.post_norm else img_a
                continue

            dep_a = self.align_depth[i](depth_features[key])
            m = m_maps[key]

            if self.prior_enabled and self.prior_use_valid_hole:
                dep_a = dep_a * priors_ms[key]["valid"]

            fused_base = m * dep_a + (1.0 - m) * img_a
            # out = fused_base + self.residual_alpha * img_a
            # 严格无放大版残差
            out = m * (dep_a + self.residual_alpha * img_a) + (1.0 - m) * img_a

            if self.post_norm:
                out = self.post_norm[i](out)
            fused[key] = out

        losses: Dict[str, torch.Tensor] = {}
        if self.training and m_maps:
            if self.loss_entropy_weight > 0:
                ent_terms = [
                    -(
                        m.clamp(1e-6, 1.0 - 1e-6) * torch.log(m.clamp(1e-6, 1.0 - 1e-6))
                        + (1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                        * torch.log(1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                    ).mean()
                    for m in m_maps.values()
                ]
                losses["loss_mgm_entropy"] = (
                    self.loss_entropy_weight * torch.stack(ent_terms).mean()
                )

            # The entropy loss correctly encourages m to be binary.
            if self.noise_mask_weight > 0 and depth_noise_mask is not None:
                bces = []
                for m in m_maps.values():
                    # 1. 上采样噪声掩码到当前特征图尺寸
                    target_noise_mask = _bilinear(
                        depth_noise_mask.float(), m.shape[-2:]
                    )
                    # 2. 对掩码做一次最大池化，将其影响范围扩大1像素
                    #    这将噪声像素的影响扩展到周围1像素邻域，实现监督信号的软化
                    #    避免模型在洞的边缘学习到过于陡峭的置信度变化，提高学习稳定性
                    #    符合物理直觉：噪声像素周围的像素也可能不可靠，创造平滑过渡带
                    #    零配置：kernel_size=3, padding=1是标准的1像素膨胀，无需调整参数
                    #    零性能开销：3x3最大池化在现代GPU上几乎是瞬时的
                    target_noise_mask_dilated = F.max_pool2d(
                        target_noise_mask, kernel_size=3, stride=1, padding=1
                    )
                    # 3. 使用膨胀后的掩码作为监督目标
                    bce = F.binary_cross_entropy(m, 1.0 - target_noise_mask_dilated)
                    bces.append(bce)

                if bces:
                    losses["loss_mgm_noise"] = (
                        self.noise_mask_weight * torch.stack(bces).mean()
                    )

        return fused, m_maps, losses


def build_mgm(cfg) -> MultiModalGatedFusion:
    """
    Builds the MultiModalGatedFusion module from a config.
    This is the correct implementation for Detectron2's registry.
    """
    return MultiModalGatedFusion(cfg)
