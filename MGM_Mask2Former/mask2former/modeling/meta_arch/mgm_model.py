import torch
import torch.nn as nn
from typing import Dict

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone, build_sem_seg_head
from detectron2.structures import ImageList


def _build_align_1x1(in_dim, out_dim):
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Conv2d(in_dim, out_dim, 1)


@META_ARCH_REGISTRY.register()
class MGMMaskFormer(nn.Module):
    """
    Minimal RGB-D MaskFormer:
      - RGB backbone: cfg.MODEL.BACKBONE.NAME (如 Swin)
      - Depth backbone: cfg.MODEL.DEPTH_BACKBONE.NAME (ConvNeXtDepthBackbone)
      - Fuse RGB/Depth multi-scale features -> feed to standard sem_seg_head
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1,1,1)
        self.pixel_std  = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1,1,1)

        # RGB backbone
        self.backbone = build_backbone(cfg)
        rgb_shapes = self.backbone.output_shape()

        # Depth backbone
        from detectron2.modeling import BACKBONE_REGISTRY
        depth_bb_name = cfg.MODEL.DEPTH_BACKBONE.NAME
        depth_bb_ctor = BACKBONE_REGISTRY.get(depth_bb_name)
        self.depth_backbone = depth_bb_ctor(cfg, input_shape=None)

        # 对齐层：把 RGB & Depth 的 C 对齐到同一 FUSION_DIM
        fdim = int(cfg.MODEL.MGM.FUSION_DIM)
        self.align_rgb  = nn.ModuleDict()
        self.align_dep  = nn.ModuleDict()
        for k, spec in rgb_shapes.items():
            self.align_rgb[k] = _build_align_1x1(spec.channels, fdim)
            self.align_dep[k] = _build_align_1x1(self.depth_backbone.output_shape()[k].channels, fdim)

        self.fusion_type = cfg.MODEL.MGM.FUSION_TYPE.lower()  # 'add' | 'concat' | 'attention'(TODO)
        out_dim_to_head = fdim if self.fusion_type!='concat' else (2*fdim)

        # 若 concat，需要再压回 head 期望的维度（通常就是 CONVS_DIM=256）
        convs_dim = int(cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM)
        self.fuse_proj = nn.ModuleDict()
        for k in rgb_shapes.keys():
            if out_dim_to_head != convs_dim:
                self.fuse_proj[k] = nn.Conv2d(out_dim_to_head, convs_dim, 1)
            else:
                self.fuse_proj[k] = nn.Identity()

        # 原版 head（pixel decoder + transformer decoder）
        # 注意：head 的 IN_FEATURES 要与传入的 dict key 匹配（res2,res3,res4,res5）
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

    def normalize(self, x):
        return (x - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)

    def _fuse(self, rgb_feats: Dict[str, torch.Tensor], dep_feats: Dict[str, torch.Tensor]):
        out = {}
        for k in rgb_feats.keys():
            r = self.align_rgb[k](rgb_feats[k])
            d = self.align_dep[k](dep_feats[k])
            if self.fusion_type == 'add':
                f = r + d
            elif self.fusion_type == 'concat':
                f = torch.cat([r, d], dim=1)
            else:
                # 预留 attention：先用 add 兜底
                f = r + d
            f = self.fuse_proj[k](f)
            out[k] = f
        return out

    def forward(self, batched_inputs):
        # RGB
        images = [self.normalize(x["image"].to(self.pixel_mean.device).float()) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        rgb_feats = self.backbone(images.tensor)

        # Depth（mgm mapper 会提供 [1,H,W] float）
        if "depth" not in batched_inputs[0]:
            raise ValueError("MGMMaskFormer expects 'depth' in batched_inputs. Check your RGB-D mapper & dataset.")
        depths = [x["depth"].to(images.tensor.device).float() for x in batched_inputs]
        # 和 images 一样 pad 到可整除
        depths = ImageList.from_tensors(depths, self.backbone.size_divisibility)
        dep_feats = self.depth_backbone(depths.tensor)

        fused = self._fuse(rgb_feats, dep_feats)
        results, losses = self.sem_seg_head(fused)

        if self.training:
            gt_instances = [x["instances"].to(images.tensor.device) for x in batched_inputs]
            losses = self.sem_seg_head.losses(results, gt_instances)
            return losses
        else:
            return self.sem_seg_head.inference(results, images.image_sizes)
