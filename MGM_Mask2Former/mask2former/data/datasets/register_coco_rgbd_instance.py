# Copyright (c) Facebook, Inc. and its affiliates.
# RGB-D COCO dataset registration (minimal; mapper will locate depth/noise by DATASET_ROOT)
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

__all__ = ["register_all_coco_rgbd"]

# 你固定的数据集结构：
# DATASET_ROOT/
#   annotations/instances_{train,val,test}.json
#   images/{train,val,test}/...png
#   depth/
#     depth_npy/{train,val,test}/...npy(.npz)                # 同名
#     depth_noise_mask/{train,val,test}/...{png/jpg/npy}     # 同名（二值）

def register_all_coco_rgbd(dataset_root: str,
                           prefix: str = "coco_instance_rgbd"):
    """
    只做一件事：把 COCO json + images/<split> 注册到 Detectron2。
    深度与噪声掩码路径完全由 mapper 按 DATASET_ROOT 自动匹配。
    """
    assert os.path.isdir(dataset_root), f"Invalid DATASET_ROOT: {dataset_root}"

    splits = {
        "train": ("annotations/instances_train.json", "images/train"),
        "val":   ("annotations/instances_val.json",   "images/val"),
        "test":  ("annotations/instances_test.json",  "images/test"),
    }

    for split, (ann_rel, img_rel) in splits.items():
        json_file = os.path.join(dataset_root, ann_rel)
        image_root = os.path.join(dataset_root, img_rel)
        name = f"{prefix}_{split}"

        register_coco_instances(name, {}, json_file, image_root)

        # 可选：把 DATASET_ROOT 也挂到元数据里（便于调试/可视化时取用）
        MetadataCatalog.get(name).set(
            evaluator_type="coco",
            dataset_root=dataset_root,
        )

    print(f"[register_all_coco_rgbd] Registered splits under prefix '{prefix}' at {dataset_root}")

