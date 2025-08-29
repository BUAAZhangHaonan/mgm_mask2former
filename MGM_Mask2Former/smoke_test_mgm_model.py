# smoke_test_mgm.py
# -*- coding: utf-8 -*-
import argparse
import random

import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BitMasks, Instances

from mask2former.modeling.config.mgm_config import add_mgm_config
from mask2former.modeling.meta_arch.mgm_model import MGMMaskFormer


def register_dummy_dataset(name: str, num_thing_classes: int = 3):
    # 空数据集占位即可；关键是给 meta 需要的映射
    if name in DatasetCatalog.list():
        return
    DatasetCatalog.register(name, lambda: [])
    # 设置 thing 映射（供 instance / panoptic 推理路径使用）
    thing_map = {i: i for i in range(num_thing_classes)}
    MetadataCatalog.get(name).set(
        thing_dataset_id_to_contiguous_id=thing_map,
        thing_classes=[f"cls{i}" for i in range(num_thing_classes)],
    )


def build_cfg(yaml_path: str, device: str = None):
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(yaml_path)

    # 覆盖成 dummy 数据集，避免依赖真实注册
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

    # 推理只开 instance，关掉 semantic/panoptic 可减少无关分支
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True

    # 避免权重加载
    cfg.MODEL.WEIGHTS = ""

    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    return cfg


def make_random_instances(h, w, num_classes, max_inst=2):
    """构造随机 Instances（BitMasks + gt_classes）"""
    n = random.randint(1, max_inst)
    masks = torch.zeros((n, h, w), dtype=torch.bool)
    for i in range(n):
        # 随机矩形作为 mask
        rh, rw = random.randint(h // 8, h // 3), random.randint(w // 8, w // 3)
        y0 = random.randint(0, max(0, h - rh))
        x0 = random.randint(0, max(0, w - rw))
        masks[i, y0 : y0 + rh, x0 : x0 + rw] = True
    gt = Instances((h, w))
    gt.gt_masks = BitMasks(masks)
    gt.gt_classes = torch.randint(low=0, high=num_classes, size=(n,), dtype=torch.int64)
    return gt


def make_fake_batch(
    B=2, H=256, W=256, with_targets=True, with_noise_mask=True, device="cpu"
):
    """
    构造一个 batched_inputs 列表，元素 shape:
      - image: [3,H,W] uint8
      - depth: [1,H,W] float32 in [0,1]
      - depth_noise_mask: [1,H,W] float32 (可选)
      - instances: detectron2 Instances (训练时必备)
      - height/width: 原始尺寸（供后处理）
    """
    batch = []
    for _ in range(B):
        img = torch.randint(0, 256, (3, H, W), dtype=torch.uint8, device=device)
        depth = torch.rand(1, H, W, dtype=torch.float32, device=device)
        sample = {
            "image": img,
            "depth": depth,
            "height": H,
            "width": W,
        }
        if with_noise_mask:
            # 注意：这里已经是 CHW（1×H×W），不要再 unsqueeze
            noise = (torch.rand(1, H, W, device=device) > 0.5).float()
            sample["depth_noise_mask"] = noise
        if with_targets:
            # num_classes 在构图后从 cfg 里拿；这里先占位，稍后补
            sample["_need_targets"] = True  # 占位标记
        batch.append(sample)
    return batch


def attach_targets(batch, num_classes):
    for s in batch:
        if s.pop("_need_targets", False):
            H, W = s["height"], s["width"]
            s["instances"] = make_random_instances(H, W, num_classes)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", required=True, help="Path to mgm_swin_convnext_tiny.yaml"
    )
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    cfg = build_cfg(args.cfg, device=args.device)
    device = cfg.MODEL.DEVICE
    H = W = args.size
    B = args.batch

    # 构建模型
    model_kwargs = MGMMaskFormer.from_config(cfg)
    model = MGMMaskFormer(**model_kwargs).to(device)
    model.train()

    # 训练路径：构造 batch（含 targets）
    batched_inputs = make_fake_batch(
        B=B, H=H, W=W, with_targets=True, with_noise_mask=True, device=device
    )
    batched_inputs = attach_targets(
        batched_inputs, num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    )

    # 前向 + 反传（只做一次，验证梯度链路）
    losses = model(batched_inputs)
    loss_sum = sum(v for v in losses.values())
    print("[train] losses:", {k: float(v.detach().cpu()) for k, v in losses.items()})
    loss_sum.backward()
    print("[train] backward ok, total loss =", float(loss_sum.detach().cpu()))

    # 推理路径：去掉 targets
    model.eval()
    infer_inputs = make_fake_batch(
        B=B, H=H, W=W, with_targets=False, with_noise_mask=True, device=device
    )
    with torch.no_grad():
        outputs = model(infer_inputs)
    print("[eval] num outputs:", len(outputs))
    for i, out in enumerate(outputs):
        keys = list(out.keys())
        print(f"[eval] sample {i} keys:", keys)
        if "instances" in out:
            inst = out["instances"]
            print(
                f"  instances: {len(inst)} masks={getattr(inst, 'pred_masks', None) is not None}"
            )

    print("Smoke test passed ✔")


if __name__ == "__main__":
    main()
