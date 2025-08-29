# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by MGM Authors
import numpy as np
from typing import Callable, Dict, List, Optional, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_
from torch.amp.autocast_mode import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.depth_position_encoding import DepthPosEncoding
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, srcs, pos_embeds):
        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
            for x in srcs
        ]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 兼容新旧 PyTorch meshgrid 接口
            try:
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                    torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                    indexing="ij",
                )
            except TypeError:
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                    torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MGMMSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        transformer_in_features: List[str],
        common_stride: int,
        dpe_enabled: bool,
    ):
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        transformer_input_shape = sorted(
            transformer_input_shape.items(), key=lambda x: x[1].stride
        )
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [
            v.stride for k, v in transformer_input_shape
        ]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                )
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                ]
            )
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.dpe_enabled = dpe_enabled
        if self.dpe_enabled:
            self.depth_pe = DepthPosEncoding(hidden_dim=conv_dim)
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim, mask_dim, kernel_size=1, stride=1, padding=0
        )
        weight_init.c2_xavier_fill(self.mask_features)
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module(f"adapter_{idx+1}", lateral_conv)
            self.add_module(f"layer_{idx+1}", output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        # ===== 记录解码器使用的特征名称顺序，避免多处重复逻辑，并锁定顺序 =====
        self.decoder_level_names = self.transformer_in_features[::-1][
            : self.maskformer_num_feature_levels
        ]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v
            for k, v in input_shape.items()
            if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = 1024
        ret["transformer_enc_layers"] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        ret["transformer_in_features"] = (
            cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        )
        ret["common_stride"] = (
            cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        )
        ret["dpe_enabled"] = cfg.MODEL.DPE.ENABLED
        return ret

    @autocast(device_type="cuda", enabled=False)
    def forward_features(
        self,
        features: Dict[str, torch.Tensor],
        *,
        depth_raw: Optional[torch.Tensor] = None,
        confidence_maps: Optional[Dict[str, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # ====== 编码器输入构建（确保 PE 作用于投影后特征）======
        srcs, pos = [], []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            proj_x = self.input_proj[idx](x)
            srcs.append(proj_x)
            # 编码器阶段无需 padding_mask（此处仅形状对齐），若后续需要可扩展
            pos.append(self.pe_layer(proj_x))
        # 编码器前向
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]
        split_size_or_sections = [
            (
                level_start_index[i + 1] - level_start_index[i]
                if i < self.transformer_num_feature_levels - 1
                else y.shape[1] - level_start_index[i]
            )
            for i in range(self.transformer_num_feature_levels)
        ]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        for i, z in enumerate(y):
            out.append(
                z.transpose(1, 2).view(
                    bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]
                )
            )

        for idx, f in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(
                out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            y = output_conv(y)
            out.append(y)

        # 选出供解码器使用的前三个尺度 (低 -> 高分辨率)
        multi_scale_features = []
        for i, o in enumerate(out):
            if i < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)

        # ====== 生成 2D 位置编码（移除 mask 传递，统一无 mask 调用）=====
        pos_2d_list = []
        for feature_level in multi_scale_features:
            pos_2d_list.append(self.pe_layer(feature_level))  # 不再传 mask

        # ===== Key 位置编码 (融合深度/置信度) =====
        pos_key_list = None
        if self.dpe_enabled and depth_raw is not None and confidence_maps is not None:
            pos_key_list = []
            decoder_feature_names = self.decoder_level_names
            # 深度位置编码（全分辨率）——对齐 dtype / device
            depth_pe_base = self.depth_pe(depth_raw, padding_mask)
            for i, feature_level in enumerate(multi_scale_features):
                target_shape = feature_level.shape[-2:]
                feature_name = decoder_feature_names[i]
                conf_map = confidence_maps.get(feature_name)
                assert conf_map is not None, f"缺少特征 '{feature_name}' 的置信度图"
                if conf_map.dim() == 3:
                    conf_map = conf_map.unsqueeze(1)
                assert conf_map.shape[1] == 1, "置信度图必须单通道"
                depth_pe_scaled = F.interpolate(
                    depth_pe_base.to(feature_level.device, feature_level.dtype),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                conf_map_scaled = F.interpolate(
                    conf_map.to(feature_level.device, feature_level.dtype),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                conf_map_clamped = conf_map_scaled.clamp(0.0, 1.0)
                # 去掉 tanh，依赖 DepthPosEncoding 内部 alpha 控制幅度
                pos_key = pos_2d_list[i] + conf_map_clamped * depth_pe_scaled
                pos_key_list.append(pos_key)
        # 形状 + device 断言
        for i, (feat, p2d) in enumerate(zip(multi_scale_features, pos_2d_list)):
            assert feat.shape[-2:] == p2d.shape[-2:], f"第{i}层: 2D位置编码形状不匹配"
            assert feat.device == p2d.device, f"第{i}层: 2D位置编码与特征 device 不一致"
        if pos_key_list is not None:
            assert len(pos_key_list) == len(pos_2d_list), "pos_key_list 长度不匹配"
            for i, (pk, p2d) in enumerate(zip(pos_key_list, pos_2d_list)):
                assert pk.shape == p2d.shape, f"第{i}层: pos_key 形状不匹配"
                assert pk.device == p2d.device, f"第{i}层: pos_key device 不匹配"
        return (
            self.mask_features(out[-1]),
            out[0],
            multi_scale_features,
            pos_2d_list,
            pos_key_list,
        )
