# -*- coding: utf-8 -*-
import os
import copy
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes

from mask2former.modeling.config import add_mgm_config


__all__ = ["COCOInstanceRGBDDatasetMapper"]


# ---------------------------
# 基础增强
# ---------------------------
def _build_geom_transforms(cfg, is_train: bool):
    """仅几何增强（RGB与Depth/Mask同步）"""
    if not is_train:
        return []
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE
    augs: List[T.Augmentation] = []
    if cfg.INPUT.RANDOM_FLIP != "none":
        augs.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    augs.extend([
        T.ResizeScale(min_scale=min_scale, max_scale=max_scale,
                      target_height=image_size, target_width=image_size),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])
    return augs


def _apply_rgb_photometric(image: np.ndarray, cfg) -> np.ndarray:
    """只对 RGB 做轻量光度增强（可选）"""
    aug_cfg = cfg.INPUT.RGB_PHOTO_AUG
    if not getattr(aug_cfg, "ENABLED", False):
        return image

    img = image.astype(np.float32)
    if getattr(aug_cfg, "BRIGHTNESS", 0) > 0:
        fac = np.random.uniform(1.0 - aug_cfg.BRIGHTNESS, 1.0 + aug_cfg.BRIGHTNESS)
        img *= fac
    if getattr(aug_cfg, "CONTRAST", 0) > 0:
        mean = img.mean()
        fac = np.random.uniform(1.0 - aug_cfg.CONTRAST, 1.0 + aug_cfg.CONTRAST)
        img = (img - mean) * fac + mean
    if getattr(aug_cfg, "SATURATION", 0) > 0 or getattr(aug_cfg, "HUE", 0) > 0:
        if aug_cfg.SATURATION > 0:
            fac = np.random.uniform(1.0 - aug_cfg.SATURATION, 1.0 + aug_cfg.SATURATION)
            mean = img.mean(axis=2, keepdims=True)
            img = (img - mean) * fac + mean
        if getattr(aug_cfg, "HUE", 0) > 0:
            shift = np.random.uniform(-aug_cfg.HUE, aug_cfg.HUE)
            img = img[..., [1, 2, 0]] * (1.0 + shift)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------------------------
# 深度与噪声掩码 工具
# ---------------------------
def _normalize_and_augment_depth(depth: np.ndarray, cfg) -> np.ndarray:
    """
    输入: HxW 或 HxWx1
    输出: HxWx1, float32
    步骤: to float32 -> scale/shift -> clip -> normalize(按要求)
    约定:
      - 若截断上下限(dmin,dmax)为(0.0,1.0): 仅做 [0,1] 截断(>1→1,<0→0,中间不动)
      - 否则: 按 minmax 线性归一化到 [0,1]，再按可选目标区间映射
    """
    # 形状统一到 HxWx1
    if depth.ndim == 2:
        depth = depth[..., None]
    elif not (depth.ndim == 3 and depth.shape[2] == 1):
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    depth = depth.astype(np.float32)

    # 1) 单位换算与平移
    depth = depth * float(cfg.INPUT.DEPTH_SCALE) + float(cfg.INPUT.DEPTH_SHIFT)

    # 2) 截断区间
    dmin = float(cfg.INPUT.DEPTH_CLIP_MIN)
    dmax = float(cfg.INPUT.DEPTH_CLIP_MAX)

    # 安全分支：无效范围则直接返回（只做了scale/shift）
    if not (dmax > dmin):
        return depth.astype(np.float32)

    # 先按 [dmin, dmax] 截断（“截断的上下限”语义）
    depth = np.clip(depth, dmin, dmax)

    # 3) 归一化规则
    if dmin == 0.0 and dmax == 1.0:
        # 仅做 [0,1] 截断：>1→1, <0→0, 中间不动
        # （上面的 clip 已经完成该逻辑）
        return depth.astype(np.float32)

    # 否则：min-max 线性归一化到 [0,1]
    norm = (depth - dmin) / (dmax - dmin + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)

    # 若 cfg.INPUT.DEPTH_NORM 指定了目标区间 [a,b]，再映射过去（保持兼容，可选）
    depth_norm = cfg.INPUT.DEPTH_NORM
    if isinstance(depth_norm, (list, tuple)) and len(depth_norm) == 2:
        a, b = float(depth_norm[0]), float(depth_norm[1])
        norm = norm * (b - a) + a
        norm = np.clip(norm, min(a, b), max(a, b))

    return norm.astype(np.float32)

def _read_depth_npy(file_name: str) -> np.ndarray:
    """读取 .npy/.npz 深度数组，返回 HxWx1 float32"""
    arr = np.load(file_name, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        first_key = sorted(arr.files)[0]
        arr = arr[first_key]
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            pass
        elif arr.shape[0] == 1:
            arr = np.transpose(arr, (1, 2, 0))
        else:
            arr = arr[..., 0:1]
    else:
        raise ValueError(f"Unexpected depth npy shape: {arr.shape}")
    return arr.astype(np.float32)


def _read_noise_mask_any(path: str) -> np.ndarray:
    """
    读取噪声掩码（图像或 npy/npz），返回 HxWx1 float32，二值 {0,1}
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[sorted(arr.files)[0]]
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[..., 0]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[..., 0]
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected noise mask shape in npy: {arr.shape}")
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = (arr > 0.5).astype(np.float32)
        return arr[..., None]
    # 图像：取 L 通道并二值化
    m = utils.read_image(path, format="L").astype(np.float32) / 255.0
    m = (m > 0.5).astype(np.float32)
    return m  # HxWx1


def _relpath_under_images(rgb_abs_path: str, dataset_root: str) -> str:
    """
    返回 images/ 下的相对路径，如 'train/foo/bar.png'。
    强制要求 rgb_abs_path 位于 DATASET_ROOT/images/ 之下，否则报错。
    """
    images_root = os.path.join(dataset_root, "images")
    try:
        rel = os.path.relpath(rgb_abs_path, images_root)
    except Exception as e:
        raise ValueError(f"RGB path -> images relpath failed: {rgb_abs_path} vs {images_root}: {e}")

    if rel.startswith(".."):
        raise ValueError(f"RGB path is not under DATASET_ROOT/images: {rgb_abs_path} (root={images_root})")
    return rel


def _rgb_to_depth_npy_path(rgb_abs_path: str, dataset_root: str) -> str:
    """DATASET_ROOT/depth/depth_npy/<split>/<same_name>.npy（或 .npz）"""
    rel = _relpath_under_images(rgb_abs_path, dataset_root)  # e.g. 'train/.../xxx.png'
    rel_no_ext = os.path.splitext(rel)[0]
    cand = os.path.join(dataset_root, "depth", "depth_npy", rel_no_ext + ".npy")
    if os.path.isfile(cand):
        return cand
    cand_npz = os.path.splitext(cand)[0] + ".npz"
    if os.path.isfile(cand_npz):
        return cand_npz
    raise FileNotFoundError(f"Depth file not found for '{rgb_abs_path}'. Tried:\n  {cand}\n  {cand_npz}")


def _rgb_to_noise_mask_path(rgb_abs_path: str, dataset_root: str) -> str:
    """DATASET_ROOT/depth/depth_noise_mask/<split>/<same_name>.(png|jpg|jpeg|bmp|npy|npz)"""
    rel = _relpath_under_images(rgb_abs_path, dataset_root)
    rel_no_ext = os.path.splitext(rel)[0]
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz")
    for e in exts:
        cand = os.path.join(dataset_root, "depth", "depth_noise_mask", rel_no_ext + e)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("")


# ---------------------------
# Mapper 本体（只依赖 INPUT.DATASET_ROOT）
# ---------------------------
class COCOInstanceRGBDDatasetMapper:
    """
    目录结构（固定）：
      DATASET_ROOT/
        annotations/instances_{train,val,test}.json
        images/{train,val,test}/...<name>.png
        depth/
          depth_npy/{train,val,test}/...<name>.npy(.npz)
          depth_noise_mask/{train,val,test}/...<name>.(png/jpg/jpeg/bmp/npy/npz)

    JSON 的 images[*].file_name 只需写文件名（注册时 image_root 指到 images/<split>）。
    """

    @configurable
    def __init__(
        self,
        *,
        is_train: bool = True,
        tfm_gens: List[T.Augmentation],
        image_format: str,
        depth_format: str = "I",  # 兼容保留，未使用
        cfg=None,
        dataset_root: str,        # 必填：只用这一项
    ):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.cfg = cfg
        self.dataset_root = dataset_root

        if not self.dataset_root or not os.path.isdir(self.dataset_root):
            raise ValueError(f"INPUT.DATASET_ROOT is invalid: {self.dataset_root}")

        # ---- 新增：噪声掩码开关 & 目录检测 ----
        nm_cfg = getattr(self.cfg.INPUT, "NOISE_MASK", None)
        self.noise_mask_enabled = True
        self.noise_mask_check_dir = True
        if nm_cfg is not None:
            self.noise_mask_enabled = bool(getattr(nm_cfg, "ENABLED", True))
            self.noise_mask_check_dir = bool(getattr(nm_cfg, "CHECK_DIR", True))

        self.noise_mask_available = False
        self.noise_mask_root = os.path.join(self.dataset_root, "depth", "depth_noise_mask")

        if self.noise_mask_enabled:
            if self.noise_mask_check_dir:
                # 若检查目录且目录不存在，则全局跳过噪声掩码
                if os.path.isdir(self.noise_mask_root):
                    self.noise_mask_available = True
                else:
                    logging.getLogger(__name__).info(
                        f"[Mapper] Noise-mask dir not found, will skip masks: {self.noise_mask_root}"
                    )
            else:
                # 不检查目录：逐样本尝试查找（可能更慢）
                self.noise_mask_available = True

        # 基本健壮性检查（可注释掉）
        for sub in ["images", os.path.join("depth", "depth_npy")]:
            p = os.path.join(self.dataset_root, sub)
            if not os.path.isdir(p):
                logging.getLogger(__name__).warning(f"[Mapper] Missing directory: {p}")

        logging.getLogger(__name__).info(
            f"[COCOInstanceRGBDDatasetMapper] Augs: {self.tfm_gens} | RGB: {self.img_format} | "
            f"DATASET_ROOT={self.dataset_root}"
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        tfm_gens = _build_geom_transforms(cfg, is_train)
        rgb_format = cfg.INPUT.FORMAT
        if rgb_format not in ["RGB", "BGR"]:
            logging.getLogger(__name__).warning(f"Invalid RGB format {rgb_format}, using RGB")
            rgb_format = "RGB"

        dataset_root = getattr(cfg.INPUT, "DATASET_ROOT", None)
        if not dataset_root:
            raise ValueError("cfg.INPUT.DATASET_ROOT must be set (absolute path to dataset root).")

        return {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": rgb_format,
            "cfg": cfg,
            "dataset_root": dataset_root,
        }

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        dataset_dict = copy.deepcopy(dataset_dict)

        # --- 读 RGB ---
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            image = _apply_rgb_photometric(image, self.cfg)

        # --- 读取 Depth (NPY/NPZ) ---
        rgb_path = dataset_dict["file_name"]  # 由 register_coco_instances 组成的绝对路径
        depth_path = _rgb_to_depth_npy_path(rgb_path, self.dataset_root)
        depth = _read_depth_npy(depth_path)
        depth = _normalize_and_augment_depth(depth, self.cfg)

        # --- 读取 Noise Mask（可选）---
        if self.noise_mask_enabled and self.noise_mask_available:
            try:
                nm_path = _rgb_to_noise_mask_path(rgb_path, self.dataset_root)
                noise_arr = _read_noise_mask_any(nm_path)  # HxWx1, float32 in {0,1}
            except FileNotFoundError:
                noise_arr = None
        else:
            noise_arr = None

        # --- 同步几何增强（RGB/Depth/Mask 一致）---
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        depth = transforms.apply_image(depth)  # 1通道 float 用线性插值

        # 噪声掩码使用最近邻
        if noise_arr is not None:
            nm_2d = noise_arr[..., 0]
            nm_2d = transforms.apply_segmentation(nm_2d)
            nm_2d = (nm_2d > 0.5).astype(np.float32)
            dataset_dict["depth_noise_mask"] = torch.as_tensor(
                np.ascontiguousarray(nm_2d[None, ...])
            )  # [1,H,W]

        # padding mask（True 表示 padding 区域）
        padding_mask = np.ones(image.shape[:2], dtype=np.float32)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]

        # --- 打包张量 ---
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        # --- 训练注释 ---
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            if instances.has("gt_masks") and len(instances) > 0:
                instances.gt_boxes = Boxes(instances.gt_masks.get_bounding_boxes())
            instances = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances

        return dataset_dict
