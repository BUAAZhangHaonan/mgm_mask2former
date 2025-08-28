import copy
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes

__all__ = ["COCOInstanceRGBDDatasetMapper"]


def _build_geom_transforms(cfg, is_train: bool):
    """
    仅几何增强（RGB与Depth同步执行）
    """
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
    """
    只对RGB做简单光度增强（按需开启）。几何增强与注释同步，不放在这里。
    """
    aug_cfg = cfg.INPUT.RGB_PHOTO_AUG
    if not aug_cfg.ENABLED:
        return image

    img = image.astype(np.float32)
    # brightness: 乘法
    if aug_cfg.BRIGHTNESS > 0:
        fac = np.random.uniform(1.0 - aug_cfg.BRIGHTNESS,
                                1.0 + aug_cfg.BRIGHTNESS)
        img *= fac
    # contrast: 围绕均值缩放
    if aug_cfg.CONTRAST > 0:
        mean = img.mean()
        fac = np.random.uniform(1.0 - aug_cfg.CONTRAST, 1.0 + aug_cfg.CONTRAST)
        img = (img - mean) * fac + mean
    # saturation/hue: 简化版（转HSV做轻量扰动）
    if aug_cfg.SATURATION > 0 or aug_cfg.HUE > 0:
        # 转到 HSV（简易实现，不引入cv2）
        # 直接略过高开销准确HSV，做一个近似饱和度缩放（按通道方差近似）
        if aug_cfg.SATURATION > 0:
            fac = np.random.uniform(
                1.0 - aug_cfg.SATURATION, 1.0 + aug_cfg.SATURATION)
            mean = img.mean(axis=2, keepdims=True)
            img = (img - mean) * fac + mean
        # hue 简易扰动（微小通道循环偏移）
        if aug_cfg.HUE > 0:
            shift = np.random.uniform(-aug_cfg.HUE, aug_cfg.HUE)
            # 环形通道偏移模拟色相变化（非常轻量的近似）
            img = img[..., [1, 2, 0]] * (1.0 + shift)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _normalize_and_augment_depth(depth: np.ndarray, cfg) -> np.ndarray:
    """
    depth 输入: HxW 或 HxWx1，uint16/float 都可。
    步骤：to float32 -> scale/shift -> clip -> norm -> (optional) noise
    输出: HxWx1, float32 in [0,1]（若 DEPTH_NORM="minmax"）
    """
    if depth.ndim == 2:
        depth = depth[..., None]
    elif depth.ndim == 3 and depth.shape[2] == 1:
        pass  # 已经是 HxWx1
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    depth = depth.astype(np.float32)

    # scale/shift（先转米，再裁剪）
    depth = depth * float(cfg.INPUT.DEPTH_SCALE) + float(cfg.INPUT.DEPTH_SHIFT)

    # clip (米)
    dmin = float(cfg.INPUT.DEPTH_CLIP_MIN)
    dmax = float(cfg.INPUT.DEPTH_CLIP_MAX)
    if dmax > dmin:
        depth = np.clip(depth, dmin, dmax)

    # norm
    depth_norm = cfg.INPUT.DEPTH_NORM
    if isinstance(depth_norm, str):
        if depth_norm.lower() == "minmax" and dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin + 1e-6)
            depth = np.clip(depth, 0.0, 1.0)
    elif isinstance(depth_norm, (list, tuple)) and len(depth_norm) == 2:
        # 归一化到指定范围 [min, max]
        target_min, target_max = float(depth_norm[0]), float(depth_norm[1])
        if dmax > dmin:
            # 先归一化到[0,1]，再映射到目标范围
            normalized = (depth - dmin) / (dmax - dmin + 1e-6)
            depth = normalized * (target_max - target_min) + target_min
            depth = np.clip(depth, target_min, target_max)

    # noise（可选）
    if cfg.INPUT.DEPTH_NOISE.ENABLED:
        if cfg.INPUT.DEPTH_NOISE.GAUSSIAN_STD > 0:
            std = float(cfg.INPUT.DEPTH_NOISE.GAUSSIAN_STD)
            depth = depth + \
                np.random.randn(*depth.shape).astype(np.float32) * std
        if cfg.INPUT.DEPTH_NOISE.SPECKLE_STD > 0:
            std = float(cfg.INPUT.DEPTH_NOISE.SPECKLE_STD)
            depth = depth * \
                (1.0 + np.random.randn(*depth.shape).astype(np.float32) * std)
        if cfg.INPUT.DEPTH_NOISE.DROP_PROB > 0:
            p = float(cfg.INPUT.DEPTH_NOISE.DROP_PROB)
            val = float(cfg.INPUT.DEPTH_NOISE.DROP_VAL)
            mask = (np.random.rand(*depth.shape) < p).astype(np.float32)
            depth = depth * (1.0 - mask) + val * mask

        # 确保噪声后仍在合理范围内
        if isinstance(depth_norm, str) and depth_norm.lower() == "minmax":
            depth = np.clip(depth, 0.0, 1.0)

    return depth.astype(np.float32)


def _read_depth_image(file_name: str, format: str = "I") -> np.ndarray:
    """
    专门用于读取深度图的函数
    Args:
        file_name: 深度图文件路径
        format: 深度图格式，支持 "I"(32位整数), "L"(8位灰度), "RGB"等
    Returns:
        np.ndarray: 深度图数组
    """
    try:
        # 使用PIL读取深度图
        with Image.open(file_name) as img:
            if format == "I":
                # 16位深度图通常保存为mode "I"
                if img.mode != "I":
                    img = img.convert("I")
                depth = np.array(img, dtype=np.uint16)
            elif format == "L":
                # 8位灰度深度图
                if img.mode != "L":
                    img = img.convert("L")
                depth = np.array(img, dtype=np.uint8)
            elif format == "F":
                # 32位浮点深度图
                if img.mode != "F":
                    img = img.convert("F")
                depth = np.array(img, dtype=np.float32)
            else:
                # 其他格式，尝试直接转换
                depth = np.array(img)

        return depth
    except Exception as e:
        # 如果PIL读取失败，尝试使用detectron2的读取方式
        logging.getLogger(__name__).warning(
            f"PIL failed to read depth image {file_name}, trying detectron2 reader: {e}"
        )
        try:
            # 对于某些特殊格式，回退到detectron2的读取方式
            return utils.read_image(file_name, format="RGB")[:, :, 0]  # 取第一个通道
        except Exception as e2:
            raise RuntimeError(f"Failed to read depth image {file_name}: {e2}")


class COCOInstanceRGBDDatasetMapper:
    """
    Detectron2-style mapper for RGB-D instance tasks (LSJ pipeline).

    这个mapper通过@configurable装饰器自动处理参数传递：
    - 在train_net_mgm.py中调用COCOInstanceRGBDDatasetMapper(cfg, True)
    - @configurable装饰器会自动调用from_config方法
    - from_config方法解析cfg并返回__init__需要的所有参数
    """

    @configurable
    def __init__(
        self,
        *,
        is_train: bool = True,
        tfm_gens: List[T.Augmentation],
        image_format: str,
        depth_format: str = "I",
        cfg=None
    ):
        """
        参数说明：
        - is_train: 是否为训练模式
        - tfm_gens: 几何变换列表（从_build_geom_transforms生成）
        - image_format: RGB图像格式（从cfg.INPUT.FORMAT获取，应该是"RGB"或"BGR"）
        - depth_format: 深度图格式（从cfg.INPUT.DEPTH_FORMAT获取）
        - cfg: 完整配置对象
        """
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.depth_format = depth_format
        self.cfg = cfg

        logging.getLogger(__name__).info(
            f"[COCOInstanceRGBDDatasetMapper] Augmentations used in "
            f"{'training' if is_train else 'inference'}: {self.tfm_gens}"
        )
        logging.getLogger(__name__).info(
            f"[COCOInstanceRGBDDatasetMapper] RGB format: {self.img_format}, "
            f"Depth format: {self.depth_format}"
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        """
        这个方法被@configurable装饰器自动调用，负责从cfg中提取参数
        """
        # 构建几何变换
        tfm_gens = _build_geom_transforms(cfg, is_train)

        # 确保RGB图像格式有效
        rgb_format = cfg.INPUT.FORMAT
        if rgb_format not in ["RGB", "BGR"]:
            logging.getLogger(__name__).warning(
                f"Invalid RGB format {rgb_format}, using RGB instead"
            )
            rgb_format = "RGB"

        # 确保深度图像格式有效
        depth_format = cfg.INPUT.DEPTH_FORMAT
        if depth_format not in ["I", "L", "F"]:
            logging.getLogger(__name__).warning(
                f"Invalid depth format {depth_format}, using I instead"
            )
            depth_format = "I"

        return {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": rgb_format,
            "depth_format": depth_format,
            "cfg": cfg,
        }

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        dataset_dict = copy.deepcopy(dataset_dict)

        # --- 读 RGB ---
        # 这里self.img_format是"RGB"或"BGR"，utils.read_image可以正确处理
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # 先做 RGB 光度增强（仅RGB；不影响几何对齐）
        if self.is_train:
            image = _apply_rgb_photometric(image, self.cfg)

        # --- 读 Depth ---
        if "depth_file_name" not in dataset_dict:
            raise ValueError(
                "RGB-D mapper requires 'depth_file_name' in dataset_dict")

        # 使用专门的深度图读取函数
        depth = _read_depth_image(
            dataset_dict["depth_file_name"], format=self.depth_format)
        depth = _normalize_and_augment_depth(depth, self.cfg)  # HxWx1, float32

        # --- 同步几何增强（RGB/Depth/Mask 一致）---
        # 几何增强产生一个 transforms 对象；后续用它变换 depth & annotations
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        depth = transforms.apply_image(depth)  # depth 作为 1通道 float 图像处理

        # padding mask（True 表示 padding 区域）
        padding_mask = np.ones(image.shape[:2], dtype=np.float32)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # --- 转 Tensor ---
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))  # uint8
        # Depth: [1,H,W] float32
        depth_chw = torch.as_tensor(np.ascontiguousarray(
            depth.transpose(2, 0, 1)))  # float32
        dataset_dict["depth"] = depth_chw
        dataset_dict["padding_mask"] = torch.as_tensor(
            np.ascontiguousarray(padding_mask))

        # 处理深度噪声掩码
        noise_mask = None
        if "depth_noise_mask_file" in dataset_dict:
            noise_mask = utils.read_image(dataset_dict["depth_noise_mask_file"], format="L").astype(
                np.float32) / 255.0  # Normalize to [0,1]
            noise_mask = transforms.apply_segmentation(
                noise_mask)  # Apply same transforms
            dataset_dict["depth_noise_mask"] = torch.as_tensor(
                np.ascontiguousarray(noise_mask.transpose(2, 0, 1))
            )

        if not self.is_train:
            # 推理时移除 annotations，但保留必要的元数据
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # --- 注释处理 ---
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format="bitmask"
            )
            # 用 mask 更新 boxes（裁剪后更准确）
            if instances.has("gt_masks") and len(instances) > 0:
                instances.gt_boxes = Boxes(
                    instances.gt_masks.get_bounding_boxes())
            instances = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances

        return dataset_dict
