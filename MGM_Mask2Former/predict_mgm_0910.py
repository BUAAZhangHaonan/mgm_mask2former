#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一与训练阶段的 RGB-D 预处理（使用 mgm mapper 的同一逻辑），
支持单图 / 目录推理，输出可视化与 JSON 结果。

1. 单张（显式深度）:
python3 MGM_Mask2Former/predict_mgm_0910.py \
  --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml \
  --weights MGM_Mask2Former/pretrained-checkpoint/0909_1024_2K_0910_40K.pth \
  --input MGM_Mask2Former/predict_test/1024/3521_7843763521_50_scene_000000_001153_v0.png \
  --depth MGM_Mask2Former/predict_test/1024/3521_7843763521_50_scene_000000_001153_v0.npy \
  --output MGM_Mask2Former/predict_test/1024/output \
  --save-json

2. 目录（按数据集结构自动找深度）:
python3 MGM_Mask2Former/predict_mgm_0910.py \
  --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml \
  --weights MGM_Mask2Former/pretrained-checkpoint/0909_512_0.12K_0909_20K.pth \
  --input-dir /path/to/custom_images \
  --dataset-root /home/fuyx/zhn/mgm_datasets/dataset_0909_512_LESS \
  --output out_dir_mode \
  --batch-size 2 \
  --amp --save-json
"""

import argparse
import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
from torch import nn

# 项目根目录
CUR_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from mask2former.modeling.config.mgm_config import add_mgm_config

# 引入 mapper 内部函数
from mask2former.data.dataset_mappers.coco_instance_rgbd_dataset_mapper import (
    _normalize_and_augment_depth,
)

# 为了深度路径推断需要部分函数（如果需要更完全复用可 import 其它）
from mask2former.data.dataset_mappers.coco_instance_rgbd_dataset_mapper import (
    _rgb_to_depth_npy_path,
    _rgb_to_depth_noise_npy_path,
    _rgb_to_noise_mask_path,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

# --------------------------------------------------
# 工具函数
# --------------------------------------------------
def build_cfg(args):
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.defrost()
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    # 覆盖 DEVICE
    cfg.MODEL.DEVICE = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    cfg.freeze()
    return cfg


def load_model(cfg) -> nn.Module:
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def simple_color_palette(n: int) -> List[tuple]:
    rng = np.random.default_rng(20240214)
    colors = []
    for _ in range(n):
        colors.append(tuple(int(x) for x in rng.integers(0, 255, size=3)))
    return colors


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    简易（非压缩）COCO 风格 RLE，列优先。
    mask: (H,W) uint8 / bool
    """
    m = np.asarray(mask, order="F").astype(np.uint8).ravel()
    counts = []
    last = 0
    run = 0
    for val in m:
        if val == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = val
    counts.append(run)
    return {"counts": counts, "size": list(mask.shape)}


# --------------------------------------------------
# 单图准备类
# --------------------------------------------------
class SingleImageMGMPreparer:
    """
    复用/模仿训练 mapper 行为:
      - 读取 RGB
      - 读取/归一化 Depth
      - 构造 content_mask
      - （可选）噪声掩码
    不做随机增强（推理保持确定性）。
    """

    def __init__(self, cfg, dataset_root: Optional[str] = None):
        self.cfg = cfg
        self.dataset_root = dataset_root
        self.format = cfg.INPUT.FORMAT  # "RGB" or "BGR"
        # 决定是否尝试按数据集目录规则查找深度
        self.enable_dataset_lookup = dataset_root is not None and dataset_root != ""

        # 噪声掩码开关逻辑
        nm_cfg = getattr(cfg.INPUT, "NOISE_MASK", None)
        self.noise_mask_enabled = False
        self.noise_mask_check_dir = True
        if nm_cfg is not None:
            self.noise_mask_enabled = bool(getattr(nm_cfg, "ENABLED", False))
            self.noise_mask_check_dir = bool(getattr(nm_cfg, "CHECK_DIR", True))
        self.noise_mask_root_valid = False
        if self.enable_dataset_lookup and self.noise_mask_enabled:
            nm_root = os.path.join(dataset_root, "depth", "depth_noise_mask")
            if (not self.noise_mask_check_dir) or os.path.isdir(nm_root):
                self.noise_mask_root_valid = True

    def _read_rgb(self, path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(path)
        if self.format == "RGB":
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr  # BGR

    def _read_depth_npy(self, path: str) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[sorted(arr.files)[0]]
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[2] != 1:
            arr = arr[..., :1]
        return arr.astype(np.float32)

    def prepare(
        self,
        rgb_path: str,
        explicit_depth_path: Optional[str] = None,
        allow_missing_depth: bool = False,
    ) -> Dict[str, Any]:
        rgb = self._read_rgb(rgb_path)
        H, W = rgb.shape[:2]

        # 获取深度路径
        depth_arr = None
        noise_mask = None
        rgb_abs = os.path.abspath(rgb_path)

        if explicit_depth_path:  # 用户直接指定
            depth_arr = self._read_depth_npy(explicit_depth_path)
        elif self.enable_dataset_lookup:
            # 尝试噪声掩码 -> 决定使用哪套深度
            use_noise_depth = False
            depth_npy_path = None
            if self.noise_mask_enabled and self.noise_mask_root_valid:
                try:
                    nm_path = _rgb_to_noise_mask_path(rgb_abs, self.dataset_root)
                    if nm_path:
                        noise_mask = self._read_noise_mask_any(nm_path)
                        depth_npy_path = _rgb_to_depth_noise_npy_path(
                            rgb_abs, self.dataset_root
                        )
                        use_noise_depth = True
                except:
                    noise_mask = None

            if depth_npy_path is None:
                # 回退普通深度
                depth_npy_path = _rgb_to_depth_npy_path(rgb_abs, self.dataset_root)

            depth_arr = self._read_depth_npy(depth_npy_path)

        if depth_arr is None:
            if allow_missing_depth:
                depth_arr = np.zeros((H, W, 1), dtype=np.float32)
            else:
                raise FileNotFoundError(
                    f"No depth found for {rgb_path}. Provide --depth or dataset_root."
                )

        # 归一化（与 mapper 共用函数）
        depth_norm = _normalize_and_augment_depth(depth_arr, self.cfg)

        # content_mask：推理阶段直接全 1（True）
        content_mask = np.ones((H, W), dtype=bool)

        # 转张量
        img_tensor = torch.as_tensor(rgb.transpose(2, 0, 1).copy(), dtype=torch.float32)
        depth_tensor = torch.as_tensor(
            depth_norm.transpose(2, 0, 1).copy(), dtype=torch.float32
        )
        content_mask_t = torch.as_tensor(content_mask.copy())

        # 噪声掩码
        sample = {
            "image": img_tensor,
            "depth": depth_tensor,
            "content_mask": content_mask_t,
            "height": H,
            "width": W,
            "file_name": rgb_path,
        }
        if noise_mask is not None:
            sample["depth_noise_mask"] = torch.as_tensor(
                noise_mask[None, ..., 0].astype(np.float32)
            )

        return sample

    def _read_noise_mask_any(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".npy", ".npz"]:
            arr = np.load(path, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[sorted(arr.files)[0]]
            if arr.ndim == 2:
                arr = arr[..., None]
            elif arr.ndim == 3 and arr.shape[2] != 1:
                arr = arr[..., :1]
        else:
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(path)
            arr = (m > 127).astype(np.float32)[..., None]
        return arr.astype(np.float32)


# --------------------------------------------------
# 推理主逻辑
# --------------------------------------------------
def run_inference(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    amp: bool,
) -> List[Dict[str, Any]]:
    device = (
        model.device
        if hasattr(model, "device")
        else torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    )
    outputs = []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.inference_mode():
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(samples)
        else:
            outputs = model(samples)
    return outputs


def visualize_and_save(
    rgb: np.ndarray,
    instances,
    save_path: str,
    vis_threshold: float,
    alpha: float = 0.5,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = rgb.copy()
    if instances is None or len(instances) == 0:
        cv2.imwrite(save_path, img[:, :, ::-1])
        return

    inst_cpu = instances.to("cpu")
    if hasattr(inst_cpu, "scores") and vis_threshold >= 0:
        keep = inst_cpu.scores >= vis_threshold
        inst_cpu = inst_cpu[keep]

    if len(inst_cpu) == 0:
        cv2.imwrite(save_path, img[:, :, ::-1])
        return

    colors = simple_color_palette(len(inst_cpu))
    if hasattr(inst_cpu, "pred_masks"):
        masks = inst_cpu.pred_masks.numpy().astype(bool)
    else:
        masks = None

    for i in range(len(inst_cpu)):
        color = colors[i]
        if masks is not None:
            m = masks[i]
            img[m] = (alpha * np.array(color) + (1 - alpha) * img[m]).astype(np.uint8)
        if hasattr(inst_cpu, "scores"):
            s = float(inst_cpu.scores[i])
            if hasattr(inst_cpu, "pred_boxes"):
                box = inst_cpu.pred_boxes[i].tensor.numpy().astype(int)[0]
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    img,
                    f"{s:.2f}",
                    (box[0], max(0, box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img[:, :, ::-1])


def predictions_to_json(
    instances, image_id: str, score_threshold: float
) -> List[Dict[str, Any]]:
    if instances is None or len(instances) == 0:
        return []
    inst = instances.to("cpu")
    results = []
    has_mask = hasattr(inst, "pred_masks")
    has_box = hasattr(inst, "pred_boxes")
    scores = getattr(inst, "scores", None)
    classes = getattr(inst, "pred_classes", None)

    for i in range(len(inst)):
        if scores is not None and scores[i] < score_threshold:
            continue
        item = {"image_id": image_id}
        if scores is not None:
            item["score"] = float(scores[i])
        if classes is not None:
            item["category_id"] = int(classes[i])
        if has_box:
            box = inst.pred_boxes[i].tensor.numpy().tolist()[0]
            item["bbox"] = box  # x1,y1,x2,y2
        if has_mask:
            mask = inst.pred_masks[i].numpy().astype(np.uint8)
            item["segmentation"] = mask_to_rle(mask)
        results.append(item)
    return results


# --------------------------------------------------
# 主执行
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser("MGM Inference (pure mgm config)")
    parser.add_argument("--config", required=True, help="mgm yaml")
    parser.add_argument("--weights", required=True, help="模型权重")
    parser.add_argument("--input", help="单张图片路径")
    parser.add_argument("--depth", help="对应单张深度 .npy/.npz (可选)")
    parser.add_argument("--input-dir", help="目录模式：读取所有图像")
    parser.add_argument("--exts", default=".png,.jpg,.jpeg", help="目录模式图像后缀")
    parser.add_argument(
        "--dataset-root", default="", help="若提供则按数据集结构推断深度"
    )
    parser.add_argument("--output", default="MGM_Mask2Former/predict_test/512/output", help="输出根目录")
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=-1,
        help="仅用于可视化过滤；<0 表示不过滤",
    )
    parser.add_argument(
        "--json-threshold", type=float, default=0.0, help="导出 JSON 时的最小得分"
    )
    parser.add_argument("--save-json", action="store_true", help="保存 JSON 预测")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", help="启用 autocast (GPU)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="仅对目录模式起作用（聚合多张一起 forward）",
    )
    parser.add_argument("--opts", nargs="+", default=[])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    vis_dir = os.path.join(args.output, "vis")
    json_dir = os.path.join(args.output, "json")
    ensure_dir(vis_dir)
    if args.save_json:
        ensure_dir(json_dir)

    cfg = build_cfg(args)
    model = load_model(cfg)

    preparer = SingleImageMGMPreparer(cfg, dataset_root=args.dataset_root)

    all_json = []

    def pack_and_run(batch_samples):
        outputs = run_inference(model, batch_samples, amp=args.amp)
        for sample, out in zip(batch_samples, outputs):
            rgb = sample["image"].cpu().numpy().transpose(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                rgb_disp = rgb
            else:
                rgb_disp = rgb  # 保持 RGB；保存时内部会转 BGR
            instances = out.get("instances", None)
            stem = Path(sample["file_name"]).stem
            out_path = os.path.join(vis_dir, f"{stem}.jpg")
            visualize_and_save(rgb_disp, instances, out_path, args.vis_threshold)
            if args.save_json:
                preds = predictions_to_json(instances, stem, args.json_threshold)
                all_json.extend(preds)

    if args.input:
        sample = preparer.prepare(
            args.input,
            explicit_depth_path=args.depth,
            allow_missing_depth=bool(args.depth is None and not args.dataset_root),
        )
        pack_and_run([sample])
    else:
        assert args.input_dir, "必须提供 --input 或 --input-dir"
        exts = {e.strip().lower() for e in args.exts.split(",")}
        img_files = [
            p
            for p in sorted(Path(args.input_dir).glob("**/*"))
            if p.suffix.lower() in exts
        ]
        if len(img_files) == 0:
            print("目录下未找到图像")
            return
        buffer = []
        for p in img_files:
            sample = preparer.prepare(
                str(p),
                explicit_depth_path=None,
                allow_missing_depth=(not args.dataset_root),
            )
            buffer.append(sample)
            if len(buffer) == args.batch_size:
                pack_and_run(buffer)
                buffer = []
        if buffer:
            pack_and_run(buffer)

    if args.save_json:
        json_path = os.path.join(json_dir, "predictions.json")
        with open(json_path, "w") as f:
            json.dump(all_json, f, indent=2)
        print(f"[INFO] JSON saved to {json_path}")

    print("[DONE] Inference complete.")


if __name__ == "__main__":
    main()
