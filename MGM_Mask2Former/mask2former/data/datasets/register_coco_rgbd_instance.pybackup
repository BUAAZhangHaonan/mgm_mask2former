# Copyright (c) Facebook, Inc. and its affiliates.
# RGB-D COCO dataset registration
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

__all__ = ["register_coco_rgbd_instances", "load_coco_rgbd_json"]


def load_coco_rgbd_json(json_file, image_root, depth_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a COCO-format dataset with RGB and depth images.
    
    Args:
        json_file (str): path to the json file in COCO instances format.
        image_root (str): directory that contains all RGB images.
        depth_root (str): directory that contains all depth images.
        dataset_name (str or None): the name of the dataset (e.g., "coco_2017_train_rgbd").
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dict format (See DATASETS.md).
    """
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name, extra_annotation_keys)
    
    # Add depth file paths to each dataset dict
    for dataset_dict in dataset_dicts:
        rgb_name = os.path.basename(dataset_dict["file_name"])
        name, _ = os.path.splitext(rgb_name)
        depth_name = f"{name}_depth.npy"
        dataset_dict["depth_file_name"] = os.path.join(depth_root, depth_name)
    
    return dataset_dicts


def register_coco_rgbd_instances(name, metadata, json_file, image_root, depth_root):
    """
    Register a dataset in COCO's json annotation format for RGB-D instance detection/segmentation.
    
    This is compatible with the standard detectron2 registration mechanism.
    
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2017_train_rgbd".
        metadata (dict): extra metadata associated with this dataset.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all RGB images.
        depth_root (str): directory which contains all depth images.
    """
    assert isinstance(name, str), f"Dataset name must be a string, got {type(name)}"
    assert isinstance(metadata, dict), f"Metadata must be a dict, got {type(metadata)}"
    
    # Register the dataset
    DatasetCatalog.register(name, lambda: load_coco_rgbd_json(json_file, image_root, depth_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, 
        image_root=image_root, 
        depth_root=depth_root,
        evaluator_type="coco", 
        **metadata
    )


# Predefined splits for RGB-D COCO datasets
# You can uncomment and modify these based on your dataset structure
_PREDEFINED_SPLITS_COCO_RGBD = {
    "coco_instance_train_rgbd": ("coco/train", "coco/depth_train", "coco/annotations/instances_train.json"),
    "coco_instance_val_rgbd":   ("coco/val",   "coco/depth_val",   "coco/annotations/instances_val.json"),
}


def register_all_coco_rgbd(root):
    """
    Register all RGB-D COCO datasets.
    
    Args:
        root (str): the root directory that contains COCO datasets.
    """
    for dataset_name, (image_dirname, depth_dirname, json_filename) in _PREDEFINED_SPLITS_COCO_RGBD.items():
        register_coco_rgbd_instances(
            dataset_name,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_filename) if "://" not in json_filename else json_filename,
            os.path.join(root, image_dirname),
            os.path.join(root, depth_dirname),
        )


# Automatically register predefined RGB-D datasets if they are defined
if _PREDEFINED_SPLITS_COCO_RGBD:
    _root = os.getenv("DETECTRON2_DATASETS", "datasets") 
    register_all_coco_rgbd(_root)