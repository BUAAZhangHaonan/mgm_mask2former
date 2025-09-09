# -*- coding: utf-8 -*-
"""
改进版 MGM Mask2Former 烟雾测试脚本：
1. 支持加载预训练权重 (--weights)
2. 支持抽取 MGM 多尺度深度置信度图并保存 (--export-conf)
3. 计算噪声区域 / 空洞区域与正常区域的置信度统计对比
4. 可加载真实 RGB / Depth 输入（--rgb / --depth），否则生成随机 batch
5. 保留原训练 smoke 流程（可通过 --no-train 跳过）
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


# --------------------------------------------------------------------------------------
# 参数统计函数
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module, detailed: bool = True) -> Dict:
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    module_stats = {}

    def format_number(num):
        return f"{num:,}"

    def get_size_mb(num_params):
        return num_params * 4 / (1024 * 1024)

    print("=" * 80)
    print("模型参数统计")
    print("=" * 80)

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count

        module_name = name.split(".")[0] if "." in name else "root"
        if module_name not in module_stats:
            module_stats[module_name] = {
                "total": 0,
                "trainable": 0,
                "frozen": 0,
                "params": [],
            }
        module_stats[module_name]["total"] += param_count
        if param.requires_grad:
            module_stats[module_name]["trainable"] += param_count
        else:
            module_stats[module_name]["frozen"] += param_count
        module_stats[module_name]["params"].append(
            {
                "name": name,
                "shape": list(param.shape),
                "params": param_count,
                "trainable": param.requires_grad,
            }
        )

    print(f"总参数数量: {format_number(total_params)}")
    print(f"可训练参数: {format_number(trainable_params)}")
    print(f"冻结参数: {format_number(frozen_params)}")
    print(f"模型大小: {get_size_mb(total_params):.2f} MB")
    print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")

    if detailed:
        print("\n" + "=" * 80)
        print("各模块参数统计")
        print("=" * 80)
        sorted_modules = sorted(
            module_stats.items(), key=lambda x: x[1]["total"], reverse=True
        )
        for module_name, stats in sorted_modules:
            print(f"\n📦 {module_name}:")
            print(
                f"  总参数: {format_number(stats['total'])} ({stats['total']/total_params*100:.2f}%)"
            )
            print(f"  可训练: {format_number(stats['trainable'])}")
            print(f"  冻结: {format_number(stats['frozen'])}")
            print(f"  大小: {get_size_mb(stats['total']):.2f} MB")

            large_params = [p for p in stats["params"] if p["params"] > 1000]
            large_params.sort(key=lambda x: x["params"], reverse=True)
            if large_params:
                print("  主要参数层 (>1K参数):")
                for param_info in large_params[:5]:
                    trainable_mark = "✓" if param_info["trainable"] else "✗"
                    print(
                        f"    {trainable_mark} {param_info['name']}: {param_info['shape']} -> {format_number(param_info['params'])}"
                    )
                if len(large_params) > 5:
                    print(f"    ... 还有 {len(large_params) - 5} 个参数层")

    print("\n" + "=" * 80)
    print("模型结构概览")
    print("=" * 80)

    def print_model_structure(model, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
        for name, child in model.named_children():
            child_params = sum(p.numel() for p in child.parameters())
            if child_params > 0:
                print(
                    f"{prefix}├─ {name}: {child.__class__.__name__} ({child_params:,} params)"
                )
                if current_depth < max_depth - 1:
                    print_model_structure(
                        child, prefix + "│  ", max_depth, current_depth + 1
                    )

    print_model_structure(model)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "model_size_mb": get_size_mb(total_params),
        "trainable_ratio": trainable_params / total_params,
        "module_stats": module_stats,
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
    register_dummy_dataset(
        train_name, num_thing_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    )
    register_dummy_dataset(
        test_name, num_thing_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    )
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
# 构造随机 / 真实 batch
# --------------------------------------------------------------------------------------
def make_random_instances(h, w, num_classes, max_inst=2):
    n = random.randint(1, max_inst)
    masks = torch.zeros((n, h, w), dtype=torch.bool)
    for i in range(n):
        rh, rw = random.randint(h // 8, h // 3), random.randint(w // 8, w // 3)
        y0 = random.randint(0, max(0, h - rh))
        x0 = random.randint(0, max(0, w - rw))
        masks[i, y0 : y0 + rh, x0 : x0 + rw] = True
    gt = Instances((h, w))
    gt.gt_masks = BitMasks(masks)
    gt.gt_classes = torch.randint(low=0, high=num_classes, size=(n,), dtype=torch.int64)
    return gt


def make_fake_batch(
    B=2,
    H=256,
    W=256,
    with_targets=True,
    with_noise_mask=True,
    device="cpu",
    depth_noise_ratio=0.5,
):
    batch = []
    for _ in range(B):
        img = torch.randint(0, 256, (3, H, W), dtype=torch.uint8, device=device)
        depth = torch.rand(1, H, W, dtype=torch.float32, device=device)
        sample = {"image": img, "depth": depth, "height": H, "width": W}

        if with_noise_mask:
            noise = (torch.rand(1, H, W, device=device) < depth_noise_ratio).float()
            sample["depth_noise_mask"] = noise  # 1 表示噪声
        if with_targets:
            sample["_need_targets"] = True
        batch.append(sample)
    return batch


def load_rgb_depth(
    rgb_path: str,
    depth_path: str,
    device: str,
    resize: Optional[int] = None,
    keep_aspect: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    depth:
      - 若为 .npy: 读取后期望为 [H,W] 或 [1,H,W]，自动归一化到[0,1]（若看起来像>1）
      - 若为图像: 自动转 float32/255
    """
    rgb = Image.open(rgb_path).convert("RGB")
    depth_ext = os.path.splitext(depth_path)[1].lower()

    if resize is not None:
        if keep_aspect:
            # 最长边=resize
            w, h = rgb.size
            scale = resize / float(max(w, h))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
        else:
            rgb = rgb.resize((resize, resize), Image.BILINEAR)

    rgb_np = np.asarray(rgb).astype("uint8")  # [H,W,3]
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).to(device)

    if depth_ext == ".npy":
        d_np = np.load(depth_path)
        if d_np.ndim == 2:
            pass
        elif d_np.ndim == 3 and d_np.shape[0] in (1, 3):
            # assume CHW
            if d_np.shape[0] == 1:
                d_np = d_np[0]
            else:
                d_np = d_np.mean(0)
        else:
            raise ValueError("Unsupported depth npy shape.")
        d_min, d_max = float(d_np.min()), float(d_np.max())
        if d_max > 1.2:  # heuristic: normalize
            d_np = (d_np - d_min) / (d_max - d_min + 1e-6)
        depth_t = torch.from_numpy(d_np).unsqueeze(0).float().to(device)
    else:
        depth_img = Image.open(depth_path)
        if resize is not None:
            if keep_aspect:
                w2, h2 = depth_img.size
                # 与 rgb 同步缩放（简单实现：直接按 rgb 当前尺寸）
                depth_img = depth_img.resize(rgb.size, Image.NEAREST)
            else:
                depth_img = depth_img.resize((resize, resize), Image.NEAREST)
        d_np = np.asarray(depth_img).astype("float32")
        if d_np.ndim == 3:
            d_np = d_np[..., 0]
        if d_np.max() > 1.2:
            d_np /= 255.0
        depth_t = torch.from_numpy(d_np).unsqueeze(0).to(device)

    return rgb_t, depth_t


def build_batch_from_files(
    rgb_path: str,
    depth_path: str,
    device: str,
    noise_mask: bool = True,
    resize: Optional[int] = None,
) -> List[Dict]:
    rgb_t, depth_t = load_rgb_depth(rgb_path, depth_path, device=device, resize=resize)
    H, W = depth_t.shape[-2:]
    sample = {
        "image": rgb_t.to(torch.uint8),
        "depth": depth_t,
        "height": H,
        "width": W,
    }
    if noise_mask:
        # 简单基于梯度或随机模拟噪声，这里随机示例
        noise = (torch.rand(1, H, W, device=device) < 0.3).float()
        sample["depth_noise_mask"] = noise
    return [sample]


# --------------------------------------------------------------------------------------
# 置信度提取与可视化
# --------------------------------------------------------------------------------------
def extract_mgm_confidence(
    model: MGMMaskFormer, batched_inputs: List[Dict], upsample: bool = True
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """
    手动复刻 MGMMaskFormer.forward 的前半部分，获取 MGM 输出的 multi-scale confidence maps。
    返回:
        per_image_conf (list, len=B): 每张图的 {scale_key: [1,h,w]}
        depth_list (list): 原始深度 (after padding) 的裁剪回原尺寸
        noise_mask_list (list or empty): 若有 depth_noise_mask
    """
    model.eval()
    device = model.device

    # -------- 构造与 forward 一致的 ImageList --------
    images = [x["image"].to(device) for x in batched_inputs]
    images = [
        (x.float() - model.pixel_mean) / model.pixel_std for x in images
    ]  # 归一化
    images_list = ImageList.from_tensors(images, model.size_divisibility)

    depths = [x["depth"].to(device) for x in batched_inputs]
    depths_list = ImageList.from_tensors(depths, model.size_divisibility)

    assert (
        images_list.tensor.shape[-2:] == depths_list.tensor.shape[-2:]
    ), "RGB 与 Depth 填充后尺寸不一致"

    depth_raw = depths_list.tensor  # [B,1,H,W]

    depth_noise_mask = None
    if "depth_noise_mask" in batched_inputs[0]:
        dm = [x["depth_noise_mask"].to(device).float() for x in batched_inputs]
        depth_noise_mask = ImageList.from_tensors(
            dm, model.size_divisibility
        ).tensor  # [B,1,H,W]

    # padding mask 构造（与 forward 一致）
    B, _, H_pad, W_pad = images_list.tensor.shape
    padding_mask = torch.zeros((B, H_pad, W_pad), dtype=torch.bool, device=device)
    for i, (h, w) in enumerate(images_list.image_sizes):
        padding_mask[i, h:, :] = True
        padding_mask[i, :, w:] = True

    # -------- 特征与 MGM --------
    with torch.no_grad():
        rgb_features = model.rgb_backbone(images_list.tensor)
        depth_features = model.depth_backbone(depths_list.tensor)
        fused_features, confidence_maps, _ = model.mgm(
            image_features=rgb_features,
            depth_features=depth_features,
            depth_raw=depth_raw,
            rgb_image=images_list.tensor,
            depth_noise_mask=depth_noise_mask,
        )

    # confidence_maps: Dict[str, Tensor]  每个: [B,1,h,w] Sigmoid+clamp 后
    # 需要按原图尺寸裁剪（去掉 padding）
    per_image_conf: List[Dict[str, torch.Tensor]] = []
    depth_list: List[torch.Tensor] = []
    noise_mask_list: List[torch.Tensor] = []

    for i, (h, w) in enumerate(images_list.image_sizes):
        img_conf = {}
        for scale, m in confidence_maps.items():
            c = m[i : i + 1, :, :h, :w].detach()  # [1,1,h,w]
            if upsample:
                c = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
            img_conf[scale] = c.squeeze(0)  # -> [1,h,w]
        per_image_conf.append(img_conf)

        depth_c = depth_raw[i : i + 1, :, :h, :w]
        depth_list.append(depth_c)

        if depth_noise_mask is not None:
            nm = depth_noise_mask[i : i + 1, :, :h, :w]
            noise_mask_list.append(nm)

    return per_image_conf, depth_list, noise_mask_list


def apply_colormap(x: torch.Tensor, cmap: str = "turbo") -> np.ndarray:
    """
    x: [H,W] in [0,1]
    返回彩色 uint8
    若无 matplotlib，则使用简易分段 colormap
    """
    x_np = x.clamp(0, 1).cpu().numpy()
    try:
        import matplotlib.cm as cm

        if hasattr(cm, cmap):
            cm_func = getattr(cm, cmap)
        else:
            cm_func = cm.get_cmap("viridis")
        colored = cm_func(x_np)[:, :, :3]  # RGBA -> RGB
        colored = (colored * 255).astype("uint8")
        return colored
    except Exception:
        # fallback: simple jet-like
        r = np.clip(1.5 - np.abs(2 * x_np - 1.0), 0, 1)
        g = np.clip(1.5 - np.abs(2 * x_np - 0.5), 0, 1)
        b = np.clip(1.5 - np.abs(2 * x_np - 0.0), 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype("uint8")


def save_confidence_maps(
    per_image_conf: List[Dict[str, torch.Tensor]],
    depths: List[torch.Tensor],
    noise_masks: List[torch.Tensor],
    out_dir: str,
    prefix: str = "sample",
):
    os.makedirs(out_dir, exist_ok=True)
    for idx, conf_dict in enumerate(per_image_conf):
        depth = depths[idx][0, 0]  # [H,W]
        depth_img = (depth.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(depth_img).save(
            os.path.join(out_dir, f"{prefix}{idx}_depth.png")
        )

        if noise_masks:
            nm = noise_masks[idx][0, 0]
            nm_img = (nm.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(nm_img).save(
                os.path.join(out_dir, f"{prefix}{idx}_noise_mask.png")
            )

        # 聚合平均置信度
        agg_list = []
        for scale, m in conf_dict.items():
            # m: [1,H,W]
            c_map = m[0]
            agg_list.append(c_map)
            colored = apply_colormap(c_map)
            Image.fromarray(colored).save(
                os.path.join(out_dir, f"{prefix}{idx}_conf_{scale}.png")
            )
        if agg_list:
            mean_conf = torch.stack(agg_list).mean(0)
            Image.fromarray(apply_colormap(mean_conf)).save(
                os.path.join(out_dir, f"{prefix}{idx}_conf_mean.png")
            )


def confidence_statistics(
    per_image_conf: List[Dict[str, torch.Tensor]],
    depths: List[torch.Tensor],
    noise_masks: List[torch.Tensor],
    z_min: float = 0.0,
    z_max: float = 1.0,
):
    """
    输出每张图：
      - 所有尺度平均置信度
      - 噪声区域 vs 非噪声区域 平均置信度
      - 空洞区域 (depth<=z_min or depth>=z_max) vs 有效区域 平均置信度
    """
    print("\n" + "=" * 80)
    print("MGM 置信度统计")
    print("=" * 80)
    for i, conf_dict in enumerate(per_image_conf):
        # 聚合
        agg = torch.stack([m[0] for m in conf_dict.values()])  # [S,H,W]
        mean_conf = agg.mean(0)  # [H,W]

        depth = depths[i][0, 0]
        hole_mask = ((depth <= z_min) | (depth >= z_max)).float()
        valid_mask = 1.0 - hole_mask

        if noise_masks:
            noise_mask = noise_masks[i][0, 0]
        else:
            noise_mask = torch.zeros_like(depth)

        def masked_mean(t, m):
            denom = m.sum()
            if denom < 1:
                return float("nan")
            return float((t * m).sum() / denom)

        overall = float(mean_conf.mean())
        conf_noise = masked_mean(mean_conf, (noise_mask > 0.5).float())
        conf_clean = masked_mean(mean_conf, (noise_mask <= 0.5).float())
        conf_hole = masked_mean(mean_conf, (hole_mask > 0.5).float())
        conf_valid = masked_mean(mean_conf, (valid_mask > 0.5).float())

        print(f"[Image {i}] overall={overall:.4f}")
        if noise_masks:
            print(
                f"  噪声区域均值: {conf_noise:.4f}  | 非噪声区域均值: {conf_clean:.4f}  (差值 clean-noise={conf_clean - conf_noise:.4f})"
            )
        print(
            f"  空洞区域均值: {conf_hole:.4f} | 有效区域均值: {conf_valid:.4f}  (差值 valid-hole={conf_valid - conf_hole:.4f})"
        )


# --------------------------------------------------------------------------------------
# 训练目标附加
# --------------------------------------------------------------------------------------
def attach_targets(batch, num_classes):
    for s in batch:
        if s.pop("_need_targets", False):
            H, W = s["height"], s["width"]
            s["instances"] = make_random_instances(H, W, num_classes)
    return batch


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml", help="Path to config yaml")
    parser.add_argument("--weights", default="MGM_Mask2Former/pretrained-checkpoint/0907_20K_DEPTH_NOISE.pth", help="预训练权重(.pth)")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--size", type=int, default=1024, help="合成数据尺寸 (正方形)")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--detailed-stats", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="跳过训练链路测试")
    parser.add_argument("--export-conf", action="store_true", help="导出多尺度置信度图")
    parser.add_argument("--out-dir", default="output/mgm_conf_out", help="输出目录")
    parser.add_argument("--rgb", default="MGM_Mask2Former/input/DEPTH_NOISE/RGB/3521_7843763521_100_scene_000000_002186_v1.png", help="真实RGB路径，可选")
    parser.add_argument("--depth", default="MGM_Mask2Former/input/DEPTH_NOISE/DEPTH/3521_7843763521_100_scene_000000_002186_v1.npy", help="真实Depth路径，可选(npy或图像)")
    parser.add_argument(
        "--resize-input", type=int, default=None, help="读取真实图像时限制最长边"
    )
    parser.add_argument(
        "--depth-noise-ratio",
        type=float,
        default=0.5,
        help="随机生成时噪声mask比例估计 (0~1)",
    )
    args = parser.parse_args()

    cfg = build_cfg(args.cfg, device=args.device)

    # 构建模型
    model_kwargs = MGMMaskFormer.from_config(cfg)
    model = MGMMaskFormer(**model_kwargs).to(cfg.MODEL.DEVICE)

    # 加载预训练权重（如果提供）
    if args.weights and os.path.isfile(args.weights):
        print(f"🔄 加载权重: {args.weights}")
        # Detectron2 兼容
        checkpointer = DetectionCheckpointer(model)
        extra = checkpointer.load(args.weights)
        print(
            f"权重加载完成 (keys={len(model.state_dict())}), extra={list(extra.keys())}"
        )
    else:
        print("⚠ 未提供有效权重路径，使用随机初始化参数。")

    # 参数统计
    print("\n🔍 正在分析模型参数...")
    param_stats = count_parameters(model, detailed=args.detailed_stats)

    # 训练 smoke（可选）
    if not args.no_train:
        model.train()
        if args.rgb and args.depth:
            batch = build_batch_from_files(
                args.rgb,
                args.depth,
                device=cfg.MODEL.DEVICE,
                noise_mask=True,
                resize=args.resize_input,
            )
            # 若只单图，但 batch>1，复制
            if args.batch > 1:
                batch = batch * args.batch
        else:
            batch = make_fake_batch(
                B=args.batch,
                H=args.size,
                W=args.size,
                with_targets=True,
                with_noise_mask=True,
                device=cfg.MODEL.DEVICE,
                depth_noise_ratio=args.depth_noise_ratio,
            )
        batch = attach_targets(batch, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        losses = model(batch)
        loss_sum = sum(v for v in losses.values())
        print(
            "[train] losses:",
            {k: float(v.detach().cpu()) for k, v in losses.items()},
        )
        loss_sum.backward()
        print("[train] backward ok, total loss =", float(loss_sum.detach().cpu()))
    else:
        print("⏭ 跳过训练链路测试 (--no-train)")

    # 推理并抽取 MGM 置信度
    model.eval()
    if args.rgb and args.depth:
        infer_inputs = build_batch_from_files(
            args.rgb,
            args.depth,
            device=cfg.MODEL.DEVICE,
            noise_mask=True,
            resize=args.resize_input,
        )
        if args.batch > 1:
            infer_inputs = infer_inputs * args.batch
    else:
        infer_inputs = make_fake_batch(
            B=args.batch,
            H=args.size,
            W=args.size,
            with_targets=False,
            with_noise_mask=True,
            device=cfg.MODEL.DEVICE,
            depth_noise_ratio=args.depth_noise_ratio,
        )

    with torch.no_grad():
        # 正常 forward (获得实例/分割等)
        outputs = model(infer_inputs)
        print("[eval] num outputs:", len(outputs))
        for i, out in enumerate(outputs):
            print(f"[eval] sample {i} keys:", list(out.keys()))
            if "instances" in out:
                inst = out["instances"]
                print(
                    f"  instances: {len(inst)} masks={getattr(inst,'pred_masks',None) is not None}"
                )

        # 提取 MGM confidence maps
        per_image_conf, depths, noise_masks = extract_mgm_confidence(
            model, infer_inputs, upsample=True
        )

    # 统计对比
    confidence_statistics(per_image_conf, depths, noise_masks, z_min=0.0, z_max=1.0)

    # 导出
    if args.export_conf:
        save_confidence_maps(
            per_image_conf, depths, noise_masks, out_dir=args.out_dir, prefix="img"
        )
        print(f"✅ 置信度图已保存到: {args.out_dir}")

    print("\n" + "=" * 80)
    print("测试完成 - 参数统计摘要")
    print("=" * 80)
    print(f"✅ 模型总参数: {param_stats['total_params']:,}")
    print(f"✅ 可训练参数: {param_stats['trainable_params']:,}")
    print(f"✅ 模型大小: {param_stats['model_size_mb']:.2f} MB")
    print("✅ 训练/推理测试:", "跳过" if args.no_train else "通过")
    print("✅ MGM 多尺度置信度提取: 已完成")
    print("Smoke test + Confidence export passed ✔")


if __name__ == "__main__":
    main()
