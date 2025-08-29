# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by MGM Authors
from typing import Tuple, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import ShapeSpec

# --- MGM MODIFICATION START ---
# Import all our custom modules
from ..backbone.swin import D2SwinTransformer
from ..backbone.convnext_depth import ConvNeXtDepthBackbone
from ..mgm.mgm import MultiModalGatedFusion, build_mgm

# --- MGM MODIFICATION END ---
from ..criterion import SetCriterion
from ..matcher import HungarianMatcher


# --- MGM MODIFICATION START ---
# Rename class and update registry
@META_ARCH_REGISTRY.register()
class MGMMaskFormer(nn.Module):
    # --- MGM MODIFICATION END ---
    @configurable
    def __init__(
        self,
        *,
        # --- MGM MODIFICATION START ---
        # Replace single backbone with rgb_backbone, depth_backbone, and mgm
        rgb_backbone: D2SwinTransformer,
        depth_backbone: ConvNeXtDepthBackbone,
        mgm: MultiModalGatedFusion,
        # --- MGM MODIFICATION END ---
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        super().__init__()
        # --- MGM MODIFICATION START ---
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.mgm = mgm
        # --- MGM MODIFICATION END ---
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.rgb_backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        # --- MGM MODIFICATION START ---
        # Manually build the backbones and MGM module
        # Create dummy input shapes to instantiate the backbones
        rgb_input_shape = ShapeSpec(channels=3)
        depth_input_shape = ShapeSpec(channels=1)
        rgb_backbone = D2SwinTransformer(cfg, rgb_input_shape)
        depth_backbone = ConvNeXtDepthBackbone(cfg, depth_input_shape)
        mgm = build_mgm(cfg)
        # The sem_seg_head's input shape is the output of the RGB backbone (since MGM preserves it)
        sem_seg_head = build_sem_seg_head(cfg, rgb_backbone.output_shape())
        # --- MGM MODIFICATION END ---

        # Criterion building is unchanged
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "rgb_backbone": rgb_backbone,
            "depth_backbone": depth_backbone,
            "mgm": mgm,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # --- MGM MODIFICATION START ---
        # Preprocess images and depths
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x.float() - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # Mapper provides 'depth' which is normalized and ready for the backbone
        depths = [x["depth"].to(self.device) for x in batched_inputs]
        depths = ImageList.from_tensors(depths, self.size_divisibility)

        # The same tensor can be used for depth_raw for prior calculation
        depth_raw = depths.tensor
        depth_noise_mask = None
        if "depth_noise_mask" in batched_inputs[0]:
            masks = [x["depth_noise_mask"].to(self.device) for x in batched_inputs]
            depth_noise_mask = ImageList.from_tensors(
                masks, self.size_divisibility
            ).tensor

        # Get features from backbones + MGM
        rgb_features = self.rgb_backbone(images.tensor)
        depth_features = self.depth_backbone(depths.tensor)

        fused_features, confidence_maps, mgm_losses = self.mgm(
            image_features=rgb_features,
            depth_features=depth_features,
            depth_raw=depth_raw,
            rgb_image=images.tensor,  # Pass original RGB for edge consistency
            depth_noise_mask=depth_noise_mask,
            is_training=self.training,
        )

        # Pass all necessary info to the head
        outputs = self.sem_seg_head(fused_features, confidence_maps, depth_raw)
        # --- MGM MODIFICATION END ---

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # Main segmentation losses
            losses = self.criterion(outputs, targets)

            # Combine with MGM's internal losses
            losses.update(mgm_losses)

            # Apply weights
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                # Note: MGM loss weights are already applied inside the MGM module
            return losses
        else:
            # Inference logic is a direct copy and should work as is
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            del outputs
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width
                        )
                    processed_results[-1]["sem_seg"] = r
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r
            return processed_results

    # prepare_targets, semantic_inference, panoptic_inference, instance_inference
    # are direct copies and should work as is.
    def prepare_targets(self, targets, images):
        # ... (exact same code) ...
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {"labels": targets_per_image.gt_classes, "masks": padded_masks}
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # ... (exact same code) ...
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # ... (exact same code) ...
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []
        current_segment_id = 0
        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
            cur_mask_ids = (cur_scores.view(-1, 1, 1) * cur_masks).argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # ... (exact same code) ...
        image_size = mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = (
                    lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
