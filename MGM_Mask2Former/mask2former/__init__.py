# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
# RGB
from .modeling.config.mask2former_config import add_maskformer2_config
# RGB-D
from .modeling.config.mgm_config import add_mgm_config

# dataset loading
# RGB
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
# RGB-D
from .data.dataset_mappers.coco_instance_rgbd_dataset_mapper import COCOInstanceRGBDDatasetMapper

# models
# RGB
from .modeling.meta_arch.maskformer_model import MaskFormer
# RGB-D
from .modeling.meta_arch.mgm_model import MGMMaskFormer
# RGB
from .utils.test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
# RGB
from .evaluation.instance_evaluation import InstanceSegEvaluator

# register
# RGB-D
import mask2former.data.datasets.register_coco_rgbd_instance
