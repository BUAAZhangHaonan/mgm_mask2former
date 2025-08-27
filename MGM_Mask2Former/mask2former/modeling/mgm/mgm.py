import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable


class DepthPriorExtractor(nn.Module):
    """Extract depth priors for confidence estimation"""

    def __init__(self):
        super().__init__()
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_gradients(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute depth gradient magnitude"""
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # Pad depth for gradient computation
        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')

        # Compute gradients
        grad_x = F.conv2d(depth_pad, self.sobel_x)
        grad_y = F.conv2d(depth_pad, self.sobel_y)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Robust normalization per image
        b = grad_mag.shape[0]
        grad_mag_flat = grad_mag.view(b, -1)

        # Use percentiles for robust normalization
        percentile_95 = torch.quantile(
            grad_mag_flat, 0.95, dim=1, keepdim=True)
        percentile_05 = torch.quantile(
            grad_mag_flat, 0.05, dim=1, keepdim=True)

        grad_mag = (grad_mag_flat - percentile_05) / \
            (percentile_95 - percentile_05 + 1e-6)
        grad_mag = grad_mag.clamp(0, 1).view_as(depth)

        return grad_mag

    def compute_variance(self, depth: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Compute local variance using box filter"""
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        b, c, h, w = depth.shape

        # Use avgpool for efficient local statistics
        padding = kernel_size // 2
        pool = nn.AvgPool2d(kernel_size, stride=1,
                            padding=padding, count_include_pad=False)

        # Compute local mean and variance
        local_mean = pool(depth)
        local_var = pool(depth ** 2) - local_mean ** 2

        # Robust normalization
        var_flat = local_var.view(b, -1)
        percentile_95 = torch.quantile(var_flat, 0.95, dim=1, keepdim=True)
        percentile_05 = torch.quantile(var_flat, 0.05, dim=1, keepdim=True)

        local_var = (var_flat - percentile_05) / \
            (percentile_95 - percentile_05 + 1e-6)
        local_var = local_var.clamp(0, 1).view(b, c, h, w)

        return local_var

    def compute_validity(self, depth: torch.Tensor, z_min: float = 0.0, z_max: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute validity mask (assuming normalized depth)"""
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        valid = ((depth > z_min) & (depth < z_max)).float()
        hole = 1.0 - valid

        return valid, hole

    def compute_edge_consistency(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Compute RGB-D edge consistency"""
        # Convert RGB to grayscale
        if rgb.shape[1] == 3:
            # RGB to grayscale conversion
            gray = 0.299 * rgb[:, 0:1] + 0.587 * \
                rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]

        # Normalize gray to [0, 1]
        gray = gray / 255.0 if gray.max() > 1.0 else gray

        # Compute edges for both
        rgb_grad = self.compute_gradients(gray)
        depth_grad = self.compute_gradients(depth)

        # Consistency score
        consistency = 1.0 - torch.abs(rgb_grad - depth_grad).clamp(0, 1)

        return consistency

    def forward(self, depth_raw: torch.Tensor, rgb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract all depth priors"""
        priors = {}

        # Basic priors
        priors['gradient'] = self.compute_gradients(depth_raw)
        priors['variance'] = self.compute_variance(depth_raw)

        valid, hole = self.compute_validity(depth_raw)
        priors['valid'] = valid
        priors['hole'] = hole

        # RGB-D consistency if RGB provided
        if rgb is not None:
            priors['edge_consistency'] = self.compute_edge_consistency(
                rgb, depth_raw)
        else:
            # Default to ones if no RGB
            priors['edge_consistency'] = torch.ones_like(depth_raw)
            if depth_raw.dim() == 3:
                priors['edge_consistency'] = priors['edge_consistency'].unsqueeze(
                    1)

        return priors


class ConfidencePredictor(nn.Module):
    """Predict confidence maps for multi-scale features"""

    def __init__(self, feature_dims: List[int], temp_init: float = 1.5,
                 clamp_min: float = 0.05, clamp_max: float = 0.95):
        super().__init__()
        self.feature_dims = feature_dims
        self.temp = nn.Parameter(torch.tensor(temp_init), requires_grad=False)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Feature alignment projections
        hidden_dim = 128
        self.proj_rgb = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.GELU()
            ) for dim in feature_dims
        ])

        self.proj_depth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.GELU()
            ) for dim in feature_dims
        ])

        # Prior projection (5 channels: gradient, variance, valid, hole, edge_consistency)
        self.proj_prior = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )

        # Shared confidence predictor across scales
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(16, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1)
        )

    def set_temperature(self, temp: float):
        """Update temperature for sigmoid"""
        self.temp.data = torch.tensor(temp)

    def forward(self, rgb_feats: Dict[str, torch.Tensor],
                depth_feats: Dict[str, torch.Tensor],
                priors_multiscale: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Predict confidence maps for each scale"""
        confidences = {}

        scale_keys = ['res2', 'res3', 'res4', 'res5']

        for i, key in enumerate(scale_keys):
            if key not in rgb_feats or key not in depth_feats:
                continue

            # Project features
            rgb_proj = self.proj_rgb[i](rgb_feats[key])
            depth_proj = self.proj_depth[i](depth_feats[key])

            # Stack priors for this scale
            prior_list = []
            for prior_name in ['gradient', 'variance', 'valid', 'hole', 'edge_consistency']:
                prior_list.append(priors_multiscale[key][prior_name])

            priors_stacked = torch.cat(prior_list, dim=1)
            prior_proj = self.proj_prior(priors_stacked)

            # Concatenate all inputs
            combined = torch.cat([rgb_proj, depth_proj, prior_proj], dim=1)

            # Predict confidence
            conf_logits = self.predictor(combined)

            # Apply temperature-controlled sigmoid with clamping
            conf = torch.sigmoid(conf_logits / self.temp)
            conf = conf.clamp(self.clamp_min, self.clamp_max)

            confidences[key] = conf

        return confidences


class MultiModalGatedFusion(nn.Module):
    """Multi-Modal Gated Fusion (MGM) module for early fusion"""

    @configurable
    def __init__(
        self,
        *,
        feature_dims: List[int],
        residual_alpha: float = 0.05,
        temp_init: float = 1.5,
        temp_final: float = 1.0,
        temp_steps: int = 5000,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
        loss_entropy_weight: float = 0.01,
        loss_var_weight: float = 0.01,
        noise_mask_weight: float = 0.1,
    ):
        super().__init__()

        self.feature_dims = feature_dims
        self.residual_alpha = residual_alpha
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.temp_steps = temp_steps
        self.loss_entropy_weight = loss_entropy_weight
        self.loss_var_weight = loss_var_weight
        self.noise_mask_weight = noise_mask_weight

        # Prior extractor
        self.prior_extractor = DepthPriorExtractor()

        # Confidence predictor
        self.confidence_predictor = ConfidencePredictor(
            feature_dims, temp_init, clamp_min, clamp_max
        )

        # Feature alignment layers
        self.align_rgb = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GroupNorm(32, dim)
            ) for dim in feature_dims
        ])

        self.align_depth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GroupNorm(32, dim)
            ) for dim in feature_dims
        ])

        self.current_step = 0

    @classmethod
    def from_config(cls, cfg):
        return {
            "feature_dims": [96, 192, 384, 768],  # Swin-T/ConvNeXt-T dims
            "residual_alpha": cfg.MODEL.MGM.RESIDUAL_ALPHA,
            "temp_init": cfg.MODEL.MGM.get("TEMP_INIT", 1.5),
            "temp_final": cfg.MODEL.MGM.get("TEMP_FINAL", 1.0),
            "temp_steps": cfg.MODEL.MGM.get("TEMP_STEPS", 5000),
            "clamp_min": cfg.MODEL.MGM.get("CLAMP_MIN", 0.05),
            "clamp_max": cfg.MODEL.MGM.get("CLAMP_MAX", 0.95),
            "loss_entropy_weight": cfg.MODEL.MGM.LOSS_ENTROPY_W,
            "loss_var_weight": cfg.MODEL.MGM.LOSS_VAR_W,
            "noise_mask_weight": cfg.MODEL.MGM.get("NOISE_MASK_WEIGHT", 0.1),
        }

    def update_temperature(self):
        """Update temperature based on training step"""
        if self.current_step < self.temp_steps:
            progress = self.current_step / self.temp_steps
            temp = self.temp_init + \
                (self.temp_final - self.temp_init) * progress
            self.confidence_predictor.set_temperature(temp)
        self.current_step += 1

    def extract_multiscale_priors(
        self,
        depth_raw: torch.Tensor,
        rgb: torch.Tensor,
        target_sizes: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract priors at multiple scales"""
        # Extract priors at original scale
        priors_orig = self.prior_extractor(depth_raw, rgb)

        # Downsample to each scale
        priors_multiscale = {}
        for key, (h, w) in target_sizes.items():
            priors_multiscale[key] = {}
            for prior_name, prior_tensor in priors_orig.items():
                # Bilinear interpolation with align_corners=False
                resized = F.interpolate(
                    prior_tensor,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )
                priors_multiscale[key][prior_name] = resized

        return priors_multiscale

    def forward(
        self,
        rgb_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for MGM
        
        Returns:
            - fused_features: Dict of fused multi-scale features
            - confidence_maps: Dict of confidence maps per scale
            - losses: Dict of MGM-specific losses
        """
        if is_training:
            self.update_temperature()

        # Get target sizes from features
        target_sizes = {
            key: (feat.shape[2], feat.shape[3])
            for key, feat in rgb_features.items()
        }

        # Extract multi-scale priors
        priors_multiscale = self.extract_multiscale_priors(
            depth_raw, rgb_image, target_sizes
        )

        # Predict confidence maps
        confidence_maps = self.confidence_predictor(
            rgb_features, depth_features, priors_multiscale
        )

        # Feature alignment and fusion
        fused_features = {}
        scale_keys = ['res2', 'res3', 'res4', 'res5']

        for i, key in enumerate(scale_keys):
            if key not in rgb_features:
                continue

            # Align features
            rgb_aligned = self.align_rgb[i](rgb_features[key])
            depth_aligned = self.align_depth[i](depth_features[key])

            # Get confidence for this scale
            m = confidence_maps[key]

            # Gated fusion with residual
            fused = m * depth_aligned + \
                (1 - m + self.residual_alpha) * rgb_aligned
            fused_features[key] = fused

        # Compute losses
        losses = {}

        if is_training:
            # Entropy loss (encourage decisive confidence)
            entropy_loss = 0
            for key, m in confidence_maps.items():
                m_flat = m.view(-1)
                entropy = -(m_flat * torch.log(m_flat + 1e-8) +
                            (1 - m_flat) * torch.log(1 - m_flat + 1e-8)).mean()
                entropy_loss += entropy
            losses['loss_mgm_entropy'] = self.loss_entropy_weight * \
                entropy_loss / len(confidence_maps)

            # Variance loss (encourage spatial variance)
            var_loss = 0
            for key, m in confidence_maps.items():
                var = m.var()
                var_loss += -var  # Negative because we want to maximize variance
            losses['loss_mgm_variance'] = self.loss_var_weight * \
                var_loss / len(confidence_maps)

            # Optional noise mask supervision
            if depth_noise_mask is not None:
                noise_loss = 0
                for key, m in confidence_maps.items():
                    # Downsample noise mask to match confidence scale
                    h, w = m.shape[2], m.shape[3]
                    noise_mask_resized = F.interpolate(
                        depth_noise_mask, size=(h, w),
                        mode='bilinear', align_corners=False
                    )

                    # Confidence should be high where noise is low
                    target_conf = 1.0 - noise_mask_resized
                    noise_loss += F.binary_cross_entropy(m, target_conf)

                losses['loss_mgm_noise'] = self.noise_mask_weight * \
                    noise_loss / len(confidence_maps)

        return fused_features, confidence_maps, losses


def build_mgm(cfg):
    """Build MGM module from config"""
    return MultiModalGatedFusion(cfg)
