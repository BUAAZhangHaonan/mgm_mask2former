# Copyright (c) Facebook, Inc. and its affiliates.
# RGB
from .coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
# RGB-D
from .coco_instance_rgbd_dataset_mapper import COCOInstanceRGBDDatasetMapper

__all__ = ["COCOInstanceRGBDDatasetMapper", "COCOInstanceNewBaselineDatasetMapper"]
