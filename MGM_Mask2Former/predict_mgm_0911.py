#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MGM 推理与评估脚本（仅使用 mgm 配置）
- 深度读取/归一化严格复用训练 mapper 的实现
- 单图/目录/整 split（train|val|test）推理，支持计算 COCO AP
- 可视化带掩码描边（类似 Mask R-CNN）
- 输出 COCO JSON（压缩 RLE）

1. 单图推理 + 计算该图 AP（自动用 dataset_root/annotations 中对应 split 的 JSON）
python3 MGM_Mask2Former/predict_mgm_0911.py \
  --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml \
  --weights MGM_Mask2Former/pretrained-checkpoint/0909_1024_2K_0909_15K.pth \
  --input MGM_Mask2Former/predict_test/1024/3521_7843763521_25_scene_000001_000001_v0.png \
  --dataset-root /home/fuyx/zhn/mgm_datasets/0909_512_0.12K \
  --output out_single \
  --compute-ap --save-json --amp
  
2. 整个训练集（train split）平均 AP
python3 MGM_Mask2Former/predict_mgm.py \
  --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml \
  --weights output/0909_0.12K/0909_10K/model_final.pth \
  --dataset-root /home/fuyx/zhn/mgm_datasets/0909_512_0.12K \
  --eval-split train \
  --output out_train_ap \
  --compute-ap --save-json --batch-size 2 --amp
  
3. 指定一个目录评估（目录在 images/train 下可自动识别 split）
python3 MGM_Mask2Former/predict_mgm.py \
  --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml \
  --weights output/0909_0.12K/0909_10K/model_final.pth \
  --input-dir /home/fuyx/zhn/mgm_datasets/0909_512_0.12K/images/train \
  --dataset-root /home/fuyx/zhn/mgm_datasets/0909_512_0.12K \
  --output out_dir_ap \
  --compute-ap --save-json --batch-size 2 --amp

"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torch import nn
from typing import List, Dict, Any, Optional

# 项目根
CUR_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from mask2former.modeling.config.mgm_config import add_mgm_config

# 复用 mapper 的深度与路径逻辑
from mask2former.data.dataset_mappers.coco_instance_rgbd_dataset_mapper import (
    _normalize_and_augment_depth,
    _rgb_to_depth_npy_path,
    _rgb_to_depth_noise_npy_path,
    _rgb_to_noise_mask_path,
)

# COCO 评估
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_util

    HAS_COCO = True
except Exception:
    HAS_COCO = False


# ------------------------------
# 基础配置 / 模型加载
# ------------------------------
def build_cfg(args):
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.defrost()
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    cfg.freeze()
    return cfg


def load_model(cfg) -> nn.Module:
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model


# ------------------------------
# I/O 工具
# ------------------------------
def read_rgb(path: str, fmt: str):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    if fmt == "RGB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr  # BGR


def read_depth_npy(path: str):
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[sorted(arr.files)[0]]
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[2] != 1:
        arr = arr[..., :1]
    return arr.astype(np.float32)


def read_noise_mask_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[sorted(arr.files)[0]]
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[2] != 1:
            arr = arr[..., :1]
        arr = (arr > 0.5).astype(np.float32)
        return arr
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.float32)[..., None]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ------------------------------
# 样本构建（与训练 mapper 对齐）
# ------------------------------
class SampleBuilder:
    def __init__(self, cfg, dataset_root: str = ""):
        self.cfg = cfg
        self.dataset_root = dataset_root.strip()
        self.enable_lookup = len(self.dataset_root) > 0
        nm_cfg = getattr(cfg.INPUT, "NOISE_MASK", None)
        self.nm_enabled = bool(getattr(nm_cfg, "ENABLED", False)) if nm_cfg else False
        self.nm_check_dir = bool(getattr(nm_cfg, "CHECK_DIR", True)) if nm_cfg else True
        self.nm_dir_ok = False
        if self.enable_lookup and self.nm_enabled:
            nm_root = os.path.join(self.dataset_root, "depth", "depth_noise_mask")
            self.nm_dir_ok = (not self.nm_check_dir) or os.path.isdir(nm_root)

    def build(self, rgb_path: str, depth_path: Optional[str]):
        rgb = read_rgb(rgb_path, self.cfg.INPUT.FORMAT)
        H, W = rgb.shape[:2]

        # 定位/读取深度
        if depth_path is not None:
            d_arr = read_depth_npy(depth_path)
            nm_arr = None
        else:
            if not self.enable_lookup:
                raise FileNotFoundError(
                    "未提供 --depth 且未指定 --dataset-root，无法定位深度。"
                )
            rgb_abs = os.path.abspath(rgb_path)
            nm_arr = None
            d_path = None
            if self.nm_enabled and self.nm_dir_ok:
                try:
                    nm_path = _rgb_to_noise_mask_path(rgb_abs, self.dataset_root)
                    if nm_path:
                        nm_arr = read_noise_mask_any(nm_path)
                        d_path = _rgb_to_depth_noise_npy_path(
                            rgb_abs, self.dataset_root
                        )
                except:
                    nm_arr = None
            if d_path is None:
                d_path = _rgb_to_depth_npy_path(rgb_abs, self.dataset_root)
            d_arr = read_depth_npy(d_path)

        # 归一化 = 训练同款
        d_norm = _normalize_and_augment_depth(d_arr, self.cfg)

        # content_mask：全 True（无 padding）
        content_mask = np.ones((H, W), dtype=bool)

        sample = {
            "image": torch.as_tensor(
                rgb.transpose(2, 0, 1).copy(), dtype=torch.float32
            ),
            "depth": torch.as_tensor(
                d_norm.transpose(2, 0, 1).copy(), dtype=torch.float32
            ),
            "content_mask": torch.as_tensor(content_mask.copy()),
            "height": H,
            "width": W,
            "file_name": rgb_path,
        }
        if nm_arr is not None:
            sample["depth_noise_mask"] = torch.as_tensor(
                nm_arr.transpose(2, 0, 1).copy(), dtype=torch.float32
            )
        return sample, rgb


# ------------------------------
# 可视化（带描边）
# ------------------------------
def draw_instances_with_outline(
    rgb: np.ndarray,
    instances,
    save_path: str,
    vis_threshold: float = -1.0,
    alpha: float = 0.5,
    outline_thickness: int = 2,
):
    """
    - 掩码半透明覆盖
    - 外围描边（findContours）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = rgb.copy()
    if instances is None or len(instances) == 0:
        cv2.imwrite(save_path, img[:, :, ::-1])
        return

    inst = instances.to("cpu")
    if hasattr(inst, "scores") and vis_threshold >= 0:
        inst = inst[inst.scores >= vis_threshold]
    if len(inst) == 0:
        cv2.imwrite(save_path, img[:, :, ::-1])
        return

    rng = np.random.default_rng(12345)
    if hasattr(inst, "pred_masks"):
        masks = inst.pred_masks.numpy().astype(np.uint8)
    else:
        masks = None

    for i in range(len(inst)):
        color = tuple(int(x) for x in rng.integers(0, 255, 3))
        if masks is not None:
            m = masks[i]
            img[m.astype(bool)] = (
                alpha * np.array(color) + (1 - alpha) * img[m.astype(bool)]
            ).astype(np.uint8)
            # 描边
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                img,
                contours,
                -1,
                color,
                thickness=outline_thickness,
                lineType=cv2.LINE_AA,
            )

        # 画框和分数（可选）
        if hasattr(inst, "pred_boxes"):
            box = inst.pred_boxes[i].tensor.numpy().astype(int)[0]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if hasattr(inst, "scores"):
            s = float(inst.scores[i])
            pt = (
                (
                    int(inst.pred_boxes[i].tensor.numpy()[0, 0])
                    if hasattr(inst, "pred_boxes")
                    else 5
                ),
                (
                    int(inst.pred_boxes[i].tensor.numpy()[0, 1])
                    if hasattr(inst, "pred_boxes")
                    else 15
                ),
            )
            cv2.putText(
                img,
                f"{s:.2f}",
                pt,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(save_path, img[:, :, ::-1])


# ------------------------------
# COCO 结果转换与评估
# ------------------------------
def encode_mask_coco(mask: np.ndarray) -> Dict[str, Any]:
    # 使用 pycocotools 压缩 RLE（列优先）
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    # json 序列化需要 counts 是 str
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def build_imageid_and_cat_mapping(coco: "COCO"):
    # 映射 file_name -> image_id
    img_map = {}
    for img in coco.dataset["images"]:
        img_map[img["file_name"]] = img["id"]
    # 类别索引映射：model 的类别索引 -> COCO 的 category_id
    cat_ids = sorted(coco.getCatIds())
    return img_map, cat_ids


def infer_split_from_path(path: str, dataset_root: str) -> Optional[str]:
    p = Path(os.path.abspath(path))
    root = Path(os.path.abspath(dataset_root))
    try:
        rel = p.relative_to(root)
    except Exception:
        return None
    parts = [str(x) for x in rel.parts]
    # 形如 images/train/xxx.png
    for i, t in enumerate(parts):
        if t == "images" and i + 1 < len(parts):
            if parts[i + 1] in ("train", "val", "test"):
                return parts[i + 1]
    return None


def collect_images_from_split(
    dataset_root: str, split: str, exts: List[str]
) -> List[str]:
    img_dir = Path(dataset_root) / "images" / split
    files = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in exts:
            files.append(str(p))
    return files


def run_model(model, samples: List[Dict[str, Any]], use_amp: bool):
    device = next(model.parameters()).device
    dtype = torch.float16 if (use_amp and device.type == "cuda") else None
    with torch.inference_mode():
        if dtype is not None:
            with torch.autocast(device_type="cuda", dtype=dtype):
                return model(samples)
        else:
            return model(samples)


def instances_to_coco_results(
    instances, image_id: int, cat_ids: List[int], score_thr: float = 0.0
):
    results = []
    if instances is None or len(instances) == 0:
        return results
    inst = instances.to("cpu")
    has_mask = hasattr(inst, "pred_masks")
    scores = getattr(inst, "scores", None)
    classes = getattr(inst, "pred_classes", None)

    for i in range(len(inst)):
        if scores is not None and float(scores[i]) < score_thr:
            continue
        item = {
            "image_id": image_id,
            "score": float(scores[i]) if scores is not None else 1.0,
        }
        # 类别映射：将模型的 class idx -> 标注中的 category_id
        if classes is not None and len(cat_ids) > 0:
            idx = int(classes[i])
            idx = min(max(idx, 0), len(cat_ids) - 1)
            item["category_id"] = int(cat_ids[idx])
        elif len(cat_ids) > 0:
            item["category_id"] = int(cat_ids[0])
        else:
            item["category_id"] = 1

        if has_mask:
            m = inst.pred_masks[i].numpy().astype(np.uint8)
            item["segmentation"] = encode_mask_coco(m)
        results.append(item)
    return results


def summarize_and_print(cocoeval: "COCOeval"):
    stats = cocoeval.stats  # [AP, AP50, AP75, APs, APm, APl]

    def f(x):
        try:
            return f"{float(x)*100:6.3f}"
        except Exception:
            return "  nan "

    print("|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |")
    print("|:------:|:------:|:------:|:------:|:------:|:-----:|")
    print(
        f"| {f(stats[0])} | {f(stats[1])} | {f(stats[2])} | {f(stats[3])} | {f(stats[4])} | {f(stats[5])} |"
    )


# ------------------------------
# 主流程
# ------------------------------
def main():
    ap = argparse.ArgumentParser("MGM inference + AP evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", help="单图路径（自动根据 dataset_root 定位深度）")
    ap.add_argument("--depth", help="单图对应 .npy/.npz（可选）")
    ap.add_argument("--input-dir", help="目录模式：遍历图片")
    ap.add_argument(
        "--dataset-root", required=True, help="数据集根目录（用于定位深度与标注）"
    )
    ap.add_argument(
        "--eval-split", choices=["train", "val", "test"], help="若提供则评估整个 split"
    )
    ap.add_argument("--output", default="inference_out")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg")
    ap.add_argument(
        "--vis-threshold",
        type=float,
        default=-1.0,
        help="仅用于可视化的二次阈值；<0 表示不过滤",
    )
    ap.add_argument("--outline-thickness", type=int, default=2, help="描边像素")
    ap.add_argument(
        "--json-threshold", type=float, default=0.0, help="导出 JSON 的分数阈值"
    )
    ap.add_argument("--save-json", action="store_true")
    ap.add_argument(
        "--compute-ap", action="store_true", help="计算 COCO AP（需要安装 pycocotools）"
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--opts", nargs="+", default=[])
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    vis_dir = os.path.join(args.output, "vis")
    json_dir = os.path.join(args.output, "json")
    ensure_dir(vis_dir)
    if args.save_json:
        ensure_dir(json_dir)

    cfg = build_cfg(args)
    model = load_model(cfg)
    builder = SampleBuilder(cfg, dataset_root=args.dataset_root)

    # COCO GT
    coco = None
    imgid_map = {}
    cat_ids = []
    split_for_eval = (
        args.eval_split if hasattr(args, "eval-split") else args.eval_split
    )

    def load_coco_for_split(split: str):
        nonlocal coco, imgid_map, cat_ids
        if not HAS_COCO:
            raise RuntimeError(
                "未安装 pycocotools，无法计算 AP。pip install pycocotools"
            )
        ann = os.path.join(args.dataset_root, "annotations", f"instances_{split}.json")
        if not os.path.isfile(ann):
            raise FileNotFoundError(ann)
        coco = COCO(ann)
        imgid_map, cat_ids = build_imageid_and_cat_mapping(coco)

    # 构建待处理图像列表
    exts = {e.strip().lower() for e in args.exts.split(",")}
    images: List[str] = []
    if args.eval_split:
        # 直接整 split
        images = collect_images_from_split(
            args.dataset_root, args.eval_split, list(exts)
        )
        load_coco_for_split(args.eval_split)
    elif args.input:
        images = [args.input]
        s = infer_split_from_path(args.input, args.dataset_root)
        if args.compute_ap and s is not None:
            load_coco_for_split(s)
        elif args.compute_ap and s is None:
            print(
                "[WARN] 无法从路径推断 split，将尝试 train/val/test 顺序匹配 file_name。"
            )
            # 先不加载，稍后动态尝试
    elif args.input_dir:
        # 目录模式：若目录在 dataset_root/images/<split> 下，可自动推断 split
        images = [
            str(p)
            for p in sorted(Path(args.input_dir).rglob("*"))
            if p.suffix.lower() in exts
        ]
        s = infer_split_from_path(args.input_dir, args.dataset_root)
        if args.compute_ap and s is not None:
            load_coco_for_split(s)
        elif args.compute_ap:
            print(
                "[WARN] 目录不在 dataset_root/images/<split> 下，计算 AP 需要 --eval-split 或可从路径推断。"
            )
    else:
        raise ValueError(
            "请提供 --input（单图）或 --input-dir（目录）或 --eval-split。"
        )

    # 若 compute_ap 但仍未加载 coco，则尝试匹配：优先 train，再 val、test
    if args.compute_ap and coco is None:
        for s in ["train", "val", "test"]:
            ann = os.path.join(args.dataset_root, "annotations", f"instances_{s}.json")
            if os.path.isfile(ann):
                coco_try = COCO(ann)
                # 通过文件名是否存在来判断
                names = set([os.path.basename(x) for x in images])
                file_names = set([im["file_name"] for im in coco_try.dataset["images"]])
                if len(names & file_names) > 0:
                    coco = coco_try
                    imgid_map, cat_ids = build_imageid_and_cat_mapping(coco)
                    split_for_eval = s
                    print(f"[INFO] 自动匹配到 split = {s}")
                    break
        if coco is None:
            print("[WARN] 未找到匹配的标注，AP 将不计算。")
            args.compute_ap = False

    # 推理与累积结果
    results_json: List[Dict[str, Any]] = []
    buf_samples, buf_rgbs, buf_paths = [], [], []
    bs = max(1, int(args.batch_size))

    def flush():
        nonlocal results_json
        if not buf_samples:
            return
        outputs = run_model(model, buf_samples, use_amp=args.amp)
        for s, rgb, out, p in zip(buf_samples, buf_rgbs, outputs, buf_paths):
            stem = Path(p).stem
            # 可视化
            draw_instances_with_outline(
                rgb,
                out.get("instances", None),
                os.path.join(vis_dir, f"{stem}.jpg"),
                vis_threshold=args.vis_threshold,
                outline_thickness=args.outline_thickness,
            )
            # 累积 JSON
            if args.save_json and args.compute_ap and coco is not None:
                file_name = os.path.basename(p)
                if file_name in imgid_map:
                    img_id = int(imgid_map[file_name])
                    results_json += instances_to_coco_results(
                        out.get("instances", None),
                        img_id,
                        cat_ids,
                        score_thr=args.json_threshold,
                    )
        buf_samples.clear()
        buf_rgbs.clear()
        buf_paths.clear()

    for img_path in images:
        sample, rgb = builder.build(img_path, args.depth if len(images) == 1 else None)
        buf_samples.append(sample)
        buf_rgbs.append(rgb)
        buf_paths.append(img_path)
        if len(buf_samples) == bs:
            flush()
    flush()

    # 保存 JSON
    if (
        args.save_json
        and args.compute_ap
        and coco is not None
        and len(results_json) > 0
    ):
        pred_json_path = os.path.join(
            json_dir, f"predictions_{split_for_eval or 'auto'}.json"
        )
        with open(pred_json_path, "w") as f:
            json.dump(results_json, f)
        print(f"[INFO] 预测 JSON 保存至: {pred_json_path}")

    # 评估 AP
    if args.compute_ap and coco is not None and len(results_json) > 0:
        coco_dt = coco.loadRes(results_json)
        cocoeval = COCOeval(coco, coco_dt, iouType="segm")
        # 限定评估的图像范围
        eval_img_ids = [
            int(imgid_map[os.path.basename(p)])
            for p in images
            if os.path.basename(p) in imgid_map
        ]
        if len(eval_img_ids) > 0:
            cocoeval.params.imgIds = eval_img_ids
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
        summarize_and_print(cocoeval)

    print("[DONE] Inference complete.")


if __name__ == "__main__":
    main()
