# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by MGM Authors
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

# --- MGM MODIFICATION START ---
# These build functions will now point to our MGM versions via the registry
from ..transformer_decoder.maskformer_transformer_decoder import (
    build_transformer_decoder,
)
from ..pixel_decoder import build_pixel_decoder

# --- MGM MODIFICATION END ---


# --- MGM MODIFICATION START ---
# Rename class and update registry
@SEM_SEG_HEADS_REGISTRY.register()
class MGMHead(nn.Module):
    # --- MGM MODIFICATION END ---
    _version = 2

    # _load_from_state_dict is a direct copy
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # ... (exact same code) ...
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif (
            cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder"
        ):
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            raise NotImplementedError(
                "Only multi_scale_pixel_decoder is supported for transformer_in_feature with MGM."
            )

        return {
            "input_shape": {
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            # build_pixel_decoder and build_transformer_decoder will use the names
            # specified in the YAML file to build our MGM versions.
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg, transformer_predictor_in_channels, mask_classification=True
            ),
        }

    # --- MGM MODIFICATION START ---
    # The forward signature is changed to accept MGM-specific inputs.
    def forward(self, features, confidence_maps, depth_raw, mask=None):
        return self.layers(features, confidence_maps, depth_raw, mask)

    def layers(self, features, confidence_maps, depth_raw, mask=None):
        # The pixel decoder now needs depth and confidence maps
        mask_features, _, multi_scale_features, pos_2d, pos_key = (
            self.pixel_decoder.forward_features(
                features, depth_raw=depth_raw, confidence_maps=confidence_maps
            )
        )

        # The transformer decoder now needs pos_2d and pos_key
        predictions = self.predictor(
            multi_scale_features, mask_features, pos_2d, pos_key, mask
        )
        return predictions

    # --- MGM MODIFICATION END ---
