# -*- coding: utf-8 -*-
"""
MGM Mask2Former 可视化脚本（推理 + 深度置信度热力图叠加）
- 真实 RGB/Depth + 预训练权重时，仅导出 img0
- 导出多尺度置信度 png/npy
- 可将置信度上色后按透明度叠加到原图上，输出覆盖热力图

python3 MGM_Mask2Former/mgm_vis_0910.py \
  --size 512 --out-dir output/mgm_conf_out/0909_512_2K/3521_7843763521_50_scene_000000_001153_v0 \
  --no-train --export-conf --overlay \
  --rgb MGM_Mask2Former/predict_test/512/3521_7843763521_50_scene_000000_001123_v0.png \
  --depth MGM_Mask2Former/predict_test/512/3521_7843763521_50_scene_000000_001123_v0.npy \
  --weights MGM_Mask2Former/pretrained-checkpoint/0909_512_2K_0910_10K.pth \
  --overlay-alpha 0.5
"""

import argparse
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BitMasks, Instances, ImageList
from detectron2.checkpoint import DetectionCheckpointer

from mask2former.modeling.config.mgm_config import add_mgm_config
from mask2former.modeling.meta_arch.mgm_model import MGMMaskFormer

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)


# -----------------------------
# 杂项工具
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fake_batch(
    B: int = 1,
    H: int = 512,
    W: int = 512,
    with_targets: bool = False,
    with_noise_mask: bool = True,
    device: str = "cuda",
    depth_noise_ratio: float = 0.3,
) -> List[Dict]:
    """创建假的批量数据用于测试"""
    batch = []
    for _ in range(B):
        img = torch.randint(0, 256, (3, H, W), dtype=torch.uint8, device=device)
        depth = torch.rand(1, H, W, device=device)
        sample = {"image": img, "depth": depth, "height": H, "width": W}
        if with_noise_mask:
            sample["depth_noise_mask"] = (
                torch.rand(1, H, W, device=device) < depth_noise_ratio
            ).float()
        if with_targets:
            sample["_need_targets"] = True
        batch.append(sample)
    return batch


def get_size_divisibility(model) -> int:
    for k in ["size_divisibility"]:
        if hasattr(model, k) and int(getattr(model, k) or 0) > 0:
            return int(getattr(model, k))
    for k in ["backbone", "pixel_decoder"]:
        m = getattr(model, k, None)
        if m is not None and hasattr(m, "size_divisibility"):
            v = int(getattr(m, "size_divisibility") or 0)
            if v > 0:
                return v
    return 32


# --------------------------------------------------------------------------------------
# 参数统计
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module, detailed: bool = True) -> Dict:
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    module_stats = {}

    def fmt(n):
        return f"{n:,}"

    def mb(n):
        return n * 4 / (1024 * 1024)

    print("=" * 80)
    print("模型参数统计")
    print("=" * 80)
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        (
            trainable_params if p.requires_grad else frozen_params
        )  # just to satisfy linter
        if p.requires_grad:
            trainable_params += n
        else:
            frozen_params += n
        top = name.split(".")[0] if "." in name else "root"
        module_stats.setdefault(
            top, {"total": 0, "trainable": 0, "frozen": 0, "params": []}
        )
        module_stats[top]["total"] += n
        (
            module_stats[top]["trainable"]
            if p.requires_grad
            else module_stats[top]["frozen"]
        )
        if p.requires_grad:
            module_stats[top]["trainable"] += n
        else:
            module_stats[top]["frozen"] += n
        module_stats[top]["params"].append(
            {
                "name": name,
                "shape": list(p.shape),
                "params": n,
                "trainable": p.requires_grad,
            }
        )

    print(f"总参数数量: {fmt(total_params)}")
    print(f"可训练参数: {fmt(trainable_params)}")
    print(f"冻结参数: {fmt(frozen_params)}")
    print(f"模型大小: {mb(total_params):.2f} MB")
    print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")

    if detailed:
        print("\n" + "=" * 80)
        print("各模块参数统计")
        print("=" * 80)
        for mod, st in sorted(
            module_stats.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            print(f"\n📦 {mod}:")
            print(f"  总参数: {fmt(st['total'])} ({st['total']/total_params*100:.2f}%)")
            print(f"  可训练: {fmt(st['trainable'])}")
            print(f"  冻结: {fmt(st['frozen'])}")
            print(f"  大小: {mb(st['total']):.2f} MB")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "model_size_mb": mb(total_params),
        "trainable_ratio": trainable_params / total_params,
    }


# --------------------------------------------------------------------------------------
# 数据注册与配置
# --------------------------------------------------------------------------------------
def register_dummy_dataset(name: str, num_thing_classes: int = 3):
    if name in DatasetCatalog.list():
        return
    DatasetCatalog.register(name, lambda: [])
    thing_map = {i: i for i in range(num_thing_classes)}
    MetadataCatalog.get(name).set(
        thing_dataset_id_to_contiguous_id=thing_map,
        thing_classes=[f"cls{i}" for i in range(num_thing_classes)],
    )


def build_cfg(yaml_path: str, device: Optional[str] = None):
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(yaml_path)
    # dummy dataset
    train_name = "__dummy_rgbd_train__"
    test_name = "__dummy_rgbd_test__"
    ncls = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    register_dummy_dataset(train_name, num_thing_classes=ncls)
    register_dummy_dataset(test_name, num_thing_classes=ncls)
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (test_name,)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    return cfg


# --------------------------------------------------------------------------------------
# 输入构建
# --------------------------------------------------------------------------------------
def load_rgb_depth(
    rgb_path: str,
    depth_path: str,
    device: str,
    resize: Optional[int] = None,
    keep_aspect: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], np.ndarray]:
    """返回: rgb_t[3,H,W uint8], depth_t[1,H,W float], (H,W), rgb_np(H,W,3)"""
    rgb = Image.open(rgb_path).convert("RGB")
    orig_size = rgb.size  # (W,H)

    # 可能 resize
    if resize is not None:
        if keep_aspect:
            w, h = rgb.size
            scale = resize / float(max(w, h))
            rgb = rgb.resize(
                (int(round(w * scale)), int(round(h * scale))), Image.BILINEAR
            )
        else:
            rgb = rgb.resize((resize, resize), Image.BILINEAR)

    rgb_np = np.asarray(rgb).astype("uint8")  # [H,W,3]
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).to(device)  # [3,H,W]
    H, W = rgb_t.shape[-2:]

    d_ext = os.path.splitext(depth_path)[1].lower()
    if d_ext == ".npy":
        d_np = np.load(depth_path)
        if d_np.ndim == 2:
            pass
        elif d_np.ndim == 3 and d_np.shape[0] in (1, 3):
            d_np = d_np[0] if d_np.shape[0] == 1 else d_np.mean(0)
        else:
            raise ValueError(f"Unsupported depth npy shape: {d_np.shape}")
        dmin, dmax = float(d_np.min()), float(d_np.max())
        if dmax > 1.2:
            d_np = (d_np - dmin) / (dmax - dmin + 1e-6)
        depth_t = torch.from_numpy(d_np).unsqueeze(0).float().to(device)  # [1,h,w]
        if depth_t.shape[-2:] != (H, W):
            depth_t = F.interpolate(
                depth_t.unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0)
    else:
        d_img = Image.open(depth_path).resize((W, H), Image.NEAREST)
        d_np = np.asarray(d_img).astype("float32")
        if d_np.ndim == 3:
            d_np = d_np[..., 0]
        if d_np.max() > 1.2:
            d_np /= 255.0
        depth_t = torch.from_numpy(d_np).unsqueeze(0).to(device)

    return rgb_t, depth_t, (H, W), rgb_np  # rgb_np 用于叠加底图


def build_batch_from_files(
    rgb_path: str,
    depth_path: str,
    device: str,
    noise_mask: bool = True,
    resize: Optional[int] = None,
) -> Tuple[List[Dict], Tuple[int, int], np.ndarray]:
    rgb_t, depth_t, hw, rgb_np = load_rgb_depth(
        rgb_path, depth_path, device=device, resize=resize
    )
    H, W = depth_t.shape[-2:]
    assert tuple(rgb_t.shape[-2:]) == (H, W)
    sample = {"image": rgb_t.to(torch.uint8), "depth": depth_t, "height": H, "width": W}
    if noise_mask:
        sample["depth_noise_mask"] = (torch.rand(1, H, W, device=device) < 0.3).float()
    return [sample], hw, rgb_np


# --------------------------------------------------------------------------------------
# 置信度提取
# --------------------------------------------------------------------------------------
def extract_mgm_confidence(
    model: MGMMaskFormer, batched_inputs: List[Dict], upsample: bool = True
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """返回: per_image_conf[{scale: [1,H,W]}], depths[[1,1,H,W]], noise_masks"""
    model.eval()
    device = (
        model.device if hasattr(model, "device") else next(model.parameters()).device
    )
    size_div = get_size_divisibility(model)

    images = [x["image"].to(device) for x in batched_inputs]
    images = [(x.float() - model.pixel_mean) / model.pixel_std for x in images]
    images_list = ImageList.from_tensors(images, size_div)

    depths = [x["depth"].to(device) for x in batched_inputs]
    depths_list = ImageList.from_tensors(depths, size_div)
    assert images_list.tensor.shape[-2:] == depths_list.tensor.shape[-2:]

    depth_raw = depths_list.tensor
    depth_noise_mask = None
    if "depth_noise_mask" in batched_inputs[0]:
        dm = [x["depth_noise_mask"].to(device).float() for x in batched_inputs]
        depth_noise_mask = ImageList.from_tensors(dm, size_div).tensor

    with torch.inference_mode():
        rgb_features = model.rgb_backbone(images_list.tensor)
        depth_features = model.depth_backbone(depths_list.tensor)
        _, confidence_maps, _ = model.mgm(
            image_features=rgb_features,
            depth_features=depth_features,
            depth_raw=depth_raw,
            rgb_image=images_list.tensor,
            depth_noise_mask=depth_noise_mask,
        )

    per_image_conf, depth_list, noise_mask_list = [], [], []
    for i, (h, w) in enumerate(images_list.image_sizes):
        img_conf = {}
        for scale, m in confidence_maps.items():
            c = m[i : i + 1, :, :h, :w].detach()  # [1,1,h,w]
            if upsample:
                c = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
            img_conf[scale] = c.squeeze(0)  # [1,h,w]
        per_image_conf.append(img_conf)
        depth_list.append(depth_raw[i : i + 1, :, :h, :w])
        if depth_noise_mask is not None:
            noise_mask_list.append(depth_noise_mask[i : i + 1, :, :h, :w])

    return per_image_conf, depth_list, noise_mask_list


# --------------------------------------------------------------------------------------
# 上色与保存
# --------------------------------------------------------------------------------------
def apply_colormap(x: torch.Tensor, cmap: str = "turbo") -> np.ndarray:
    x_np = x.clamp(0, 1).cpu().numpy()
    try:
        import matplotlib.cm as cm

        cm_func = getattr(cm, cmap) if hasattr(cm, cmap) else cm.get_cmap("viridis")
        colored = cm_func(x_np)[:, :, :3]
        return (colored * 255).astype("uint8")
    except Exception:
        # 简易 fallback
        r = np.clip(1.5 - np.abs(2 * x_np - 1.0), 0, 1)
        g = np.clip(1.5 - np.abs(2 * x_np - 0.5), 0, 1)
        b = np.clip(1.5 - np.abs(2 * x_np - 0.0), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype("uint8")


def save_confidence_and_overlays(
    per_image_conf: List[Dict[str, torch.Tensor]],
    depths: List[torch.Tensor],
    noise_masks: List[torch.Tensor],
    base_rgb_np: np.ndarray,  # 读取后的 RGB（H,W,3）
    out_dir: str,
    prefix: str = "img0",
    cmap: str = "turbo",
    save_npy: bool = False,
    do_overlay: bool = False,
    overlay_alpha: float = 0.45,
    overlay_inplace_dir: Optional[str] = None,  # 若提供，则叠加图存到该目录
):
    os.makedirs(out_dir, exist_ok=True)

    # 仅处理第一张图（img0）
    conf_dict = per_image_conf[0]
    depth = depths[0][0, 0]  # [H,W]

    # 保存原始深度与噪声掩码（如有）
    Image.fromarray((depth.clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(
        os.path.join(out_dir, f"{prefix}_depth.png")
    )
    if noise_masks:
        nm = noise_masks[0][0, 0]
        Image.fromarray((nm.clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(
            os.path.join(out_dir, f"{prefix}_noise_mask.png")
        )

    # 计算 mean
    scales = list(conf_dict.keys())
    agg = torch.stack([conf_dict[s][0] for s in scales])  # [S,H,W]
    conf_mean = agg.mean(0)  # [H,W]

    # 保存每个尺度 + mean 的彩色热力图
    def _save_one(name: str, map2d: torch.Tensor):
        colored = apply_colormap(map2d, cmap=cmap)  # [H,W,3] uint8
        Image.fromarray(colored).save(
            os.path.join(out_dir, f"{prefix}_conf_{name}.png")
        )
        if save_npy:
            np.save(
                os.path.join(out_dir, f"{prefix}_conf_{name}.npy"), map2d.cpu().numpy()
            )
        # 叠加到原图（按当前读取尺寸作为“原始”）
        if do_overlay:
            H0, W0 = base_rgb_np.shape[0], base_rgb_np.shape[1]
            # resize 彩色热力图到 base 尺寸
            hm_img = Image.fromarray(colored).resize((W0, H0), Image.BILINEAR)
            # alpha 采用：map2d resize 后的强度 × 全局透明度
            map_resized = F.interpolate(
                map2d.unsqueeze(0).unsqueeze(0),
                size=(H0, W0),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            alpha_mask = (
                map_resized.clamp(0, 1).cpu().numpy() * 255.0 * float(overlay_alpha)
            ).astype(
                "uint8"
            )  # [H0,W0]
            hm_rgba = np.dstack([np.array(hm_img), alpha_mask])  # [H0,W0,4]
            base_rgba = np.dstack([base_rgb_np, np.full((H0, W0), 255, dtype=np.uint8)])
            out_rgba = Image.alpha_composite(
                Image.fromarray(base_rgba), Image.fromarray(hm_rgba)
            ).convert("RGB")
            save_dir = overlay_inplace_dir if overlay_inplace_dir else out_dir
            os.makedirs(save_dir, exist_ok=True)
            out_rgba.save(os.path.join(save_dir, f"{prefix}_overlay_conf_{name}.png"))

    # mean + 各尺度
    _save_one("mean", conf_mean)
    for s in scales:
        _save_one(s, conf_dict[s][0])


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------
def confidence_statistics(
    per_image_conf: List[Dict[str, torch.Tensor]],
    depths: List[torch.Tensor],
    noise_masks: List[torch.Tensor],
    z_min: float = 0.0,
    z_max: float = 1.0,
):
    print("\n" + "=" * 80)
    print("MGM 置信度统计")
    print("=" * 80)
    conf_dict = per_image_conf[0]
    agg = torch.stack([m[0] for m in conf_dict.values()])  # [S,H,W]
    mean_conf = agg.mean(0)

    depth = depths[0][0, 0]
    hole = ((depth <= z_min) | (depth >= z_max)).float()
    valid = 1.0 - hole
    noise_mask = noise_masks[0][0, 0] if noise_masks else torch.zeros_like(depth)

    def mmean(t, m):
        d = m.sum()
        return float((t * m).sum() / d) if d >= 1 else float("nan")

    overall = float(mean_conf.mean())
    conf_noise = mmean(mean_conf, (noise_mask > 0.5).float())
    conf_clean = mmean(mean_conf, (noise_mask <= 0.5).float())
    conf_hole = mmean(mean_conf, (hole > 0.5).float())
    conf_valid = mmean(mean_conf, (valid > 0.5).float())
    print(f"[img0] overall={overall:.4f}")
    if noise_masks:
        print(
            f"  噪声: {conf_noise:.4f} | 干净: {conf_clean:.4f} | Δ={conf_clean-conf_noise:.4f}"
        )
    print(
        f"  空洞: {conf_hole:.4f} | 有效: {conf_valid:.4f} | Δ={conf_valid-conf_hole:.4f}"
    )


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default="MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml"
    )
    parser.add_argument("--weights", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed-stats", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--export-conf", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--conf-cmap", default="turbo")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--rgb", default="")
    parser.add_argument("--depth", default="")
    parser.add_argument("--resize-input", type=int, default=None)
    parser.add_argument("--depth-noise-ratio", type=float, default=0.5)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1.0)
    # 叠加相关
    parser.add_argument(
        "--overlay", action="store_true", help="生成叠加到原图的可视化热力图"
    )
    parser.add_argument(
        "--overlay-alpha", type=float, default=0.45, help="叠加透明度全局系数(0~1)"
    )
    parser.add_argument(
        "--overlay-inplace", action="store_true", help="叠加图保存到原始RGB所在目录"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = build_cfg(args.cfg, device=args.device)

    # 构建模型（from_config 不可用则回退）
    try:
        model_kwargs = MGMMaskFormer.from_config(cfg)
        model = MGMMaskFormer(**model_kwargs).to(cfg.MODEL.DEVICE)
        print("✅ 使用 MGMMaskFormer.from_config 构建模型")
    except Exception as e:
        print(f"⚠ from_config 不可用或失败（{e}），回退 build_model(cfg)")
        from detectron2.modeling import build_model

        model = build_model(cfg).to(cfg.MODEL.DEVICE)

    # 权重
    has_weights = bool(args.weights and os.path.isfile(args.weights))
    if has_weights:
        print(f"🔄 加载权重: {args.weights}")
        ret = DetectionCheckpointer(model).load(args.weights)
        print(
            f"权重加载完成 | missing={len(ret.get('missing_keys',[]))} unexpected={len(ret.get('unexpected_keys',[]))}"
        )

    # 真实输入：若提供 RGB/Depth 且有权重，则强制只跑单样本 img0
    force_single = args.rgb and args.depth and has_weights

    # 参数统计
    print("\n🔍 正在分析模型参数...")
    _ = count_parameters(model, detailed=args.detailed_stats)

    # 训练 smoke（可选，且仅在非 force_single 情况下有意义）
    if (not args.no_train) and (not force_single):
        model.train()
        batch = make_fake_batch(
            B=args.batch,
            H=args.size,
            W=args.size,
            with_targets=True,
            with_noise_mask=True,
            device=cfg.MODEL.DEVICE,
            depth_noise_ratio=args.depth_noise_ratio,
        )
        # 由 mask 显式推导 gt_boxes（仅 smoke 稳定性）
        for s in batch:
            if s.get("_need_targets", False):
                H, W = s["height"], s["width"]
                n = random.randint(1, 2)
                masks = torch.zeros((n, H, W), dtype=torch.bool)
                for i in range(n):
                    rh, rw = random.randint(H // 8, H // 3), random.randint(
                        W // 8, W // 3
                    )
                    y0 = random.randint(0, max(0, H - rh))
                    x0 = random.randint(0, max(0, W - rw))
                    masks[i, y0 : y0 + rh, x0 : x0 + rw] = True
                gt = Instances((H, W))
                gt.gt_masks = BitMasks(masks)
                gt.gt_classes = torch.randint(
                    0, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, (n,), dtype=torch.int64
                )
                gt.gt_boxes = gt.gt_masks.get_bounding_boxes()
                s["instances"] = gt
                s.pop("_need_targets")
        losses = model(batch)
        loss_sum = sum(v for v in losses.values())
        print(
            "[train] losses:", {k: float(v.detach().cpu()) for k, v in losses.items()}
        )
        loss_sum.backward()
        print("[train] backward ok, total loss =", float(loss_sum.detach().cpu()))
    else:
        print("⏭ 跳过训练链路测试")

    # 推理 + 抽取
    model.eval()
    if force_single:
        infer_inputs, hw, base_rgb_np = build_batch_from_files(
            args.rgb,
            args.depth,
            device=cfg.MODEL.DEVICE,
            noise_mask=True,
            resize=args.resize_input,
        )
    else:
        base_rgb_np = None
        infer_inputs = make_fake_batch(
            B=args.batch,
            H=args.size,
            W=args.size,
            with_targets=False,
            with_noise_mask=True,
            device=cfg.MODEL.DEVICE,
            depth_noise_ratio=args.depth_noise_ratio,
        )

    with torch.inference_mode():
        outputs = model(infer_inputs)
        print("[eval] num outputs:", len(outputs))
        if outputs and isinstance(outputs, list) and "instances" in outputs[0]:
            inst = outputs[0]["instances"]
            n = len(inst) if hasattr(inst, "__len__") else -1
            print(f"[eval] img0 instances: {n}")

        per_image_conf, depths, noise_masks = extract_mgm_confidence(
            model, infer_inputs, upsample=True
        )

    # 统计只报 img0
    confidence_statistics(
        per_image_conf[:1],
        depths[:1],
        noise_masks[:1] if noise_masks else [],
        z_min=args.z_min,
        z_max=args.z_max,
    )

    # 导出
    if args.export_conf:
        # 叠加输出路径策略
        overlay_inplace_dir = None
        if args.overlay and args.overlay_inplace and force_single:
            overlay_inplace_dir = os.path.dirname(os.path.abspath(args.rgb))
        save_confidence_and_overlays(
            per_image_conf[:1],
            depths[:1],
            noise_masks[:1] if noise_masks else [],
            base_rgb_np=(
                base_rgb_np
                if base_rgb_np is not None
                else (np.zeros((args.size, args.size, 3), dtype=np.uint8))
            ),
            out_dir=args.out_dir,
            prefix="img0",
            cmap=args.conf_cmap,
            save_npy=args.save_npy,
            do_overlay=args.overlay,
            overlay_alpha=args.overlay_alpha,
            overlay_inplace_dir=overlay_inplace_dir,
        )
        where = overlay_inplace_dir if overlay_inplace_dir else args.out_dir
        print(f"✅ 置信度图/叠加图已保存到: {where}")

    print("\n完成 ✔")


if __name__ == "__main__":
    main()
