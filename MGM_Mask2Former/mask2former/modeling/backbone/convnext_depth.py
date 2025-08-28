import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath

from detectron2.modeling.backbone import Backbone
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. """

    def __init__(self, normalized_shape, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    """ConvNeXt Block"""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


@BACKBONE_REGISTRY.register()
class ConvNeXtDepthBackbone(Backbone):
    """ConvNeXt for depth input, compatible with Mask2Former"""

    def __init__(self, cfg=None, input_shape=None):
        super().__init__()

        # Get config values
        if cfg is not None and hasattr(cfg.MODEL, "DEPTH_BACKBONE"):
            depth_cfg = cfg.MODEL.DEPTH_BACKBONE.CONVNEXT
            in_chans = 1  # Depth is single channel
            depths = list(depth_cfg.DEPTHS)
            dims = list(depth_cfg.DIMS)
            drop_path_rate = float(depth_cfg.DROP_PATH_RATE)
            layer_scale_init_value = float(depth_cfg.LAYER_SCALE)
        else:
            # Default values
            in_chans = 1
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
            drop_path_rate = 0.
            layer_scale_init_value = 1e-6

        # Stem + downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Feature info for detectron2
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": dims[0], "res3": dims[1], "res4": dims[2], "res5": dims[3]
        }

        # Initialize weights
        self.apply(self._init_weights)

        # Load pretrained weights if specified
        if cfg is not None and hasattr(cfg.MODEL, "DEPTH_BACKBONE"):
            weights_path = cfg.MODEL.DEPTH_BACKBONE.WEIGHTS
            if weights_path:
                self._load_weights(weights_path)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _load_weights(self, weights_path: str):
        """Load pretrained weights"""
        print(f"[ConvNeXtDepthBackbone] Loading weights from: {weights_path}")
        try:
            checkpoint = torch.load(
                weights_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Handle potential prefix in state dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_state_dict[k.replace("backbone.", "")] = v
                else:
                    new_state_dict[k] = v

            missing, unexpected = self.load_state_dict(
                new_state_dict, strict=False)
            print(
                f"[ConvNeXtDepthBackbone] Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")

            if missing:
                print(
                    f"[ConvNeXtDepthBackbone] Missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(
                    f"[ConvNeXtDepthBackbone] Unexpected keys (first 5): {unexpected[:5]}")

        except Exception as e:
            print(
                f"[ConvNeXtDepthBackbone] Warning: Failed to load weights from {weights_path}: {e}")

    def forward(self, x):
        """Forward pass returning multi-scale features"""
        outputs = {}

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs[f"res{i+2}"] = x

        return outputs

    def output_shape(self):
        """Return feature shapes for Detectron2"""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
