import argparse
import os
import sys

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# 导入 MGM 的特定配置函数
from mask2former.modeling.config.mgm_config import add_mgm_config


class RGBDPredictor:
    """
    为 MGM 模型定制的预测器，分别处理 RGB 和 Depth 输入。
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def __call__(self, rgb_image, depth_image):
        """
        Args:
            rgb_image (np.ndarray): HxWxC 的 RGB 图像 (C=3).
            depth_image (np.ndarray): HxWxC 的 Depth 图像 (C=1).
        Returns:
            predictions (dict): 模型输出.
        """
        with torch.no_grad():
            height, width = rgb_image.shape[:2]
            # 将 RGB 和 Depth 分别转换为 Tensor
            rgb_tensor = torch.as_tensor(rgb_image.astype("float32").transpose(2, 0, 1))
            depth_tensor = torch.as_tensor(
                depth_image.astype("float32").transpose(2, 0, 1)
            )

            # 严格按照模型要求构建输入字典
            inputs = {
                "image": rgb_tensor,
                "depth": depth_tensor,
                "height": height,
                "width": width,
            }

            predictions = self.model([inputs])[0]
            return predictions


def setup_cfg(config_path, weight_path):
    cfg = get_cfg()
    # 使用 add_mgm_config 来支持 MGM 的配置项
    add_mgm_config(cfg)
    cfg.merge_from_file(config_path)

    cfg.defrost()  # 解冻配置以允许修改
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()  # 修改完成后重新冻结

    return cfg


def load_image_depth(image_path, input_type):
    # 读取 RGB 图像
    rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 强制读取为3通道BGR
    if rgb_image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 转换为RGB

    depth_image = None
    if input_type == "rgbd":
        # 读取同名的 .npy 文件作为深度图
        depth_path = os.path.splitext(image_path)[0] + ".npy"
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"找不到 depth 文件: {depth_path}")
        depth_image = np.load(depth_path)

        # 确保深度图是单通道
        if depth_image.ndim == 2:
            depth_image = np.expand_dims(depth_image, axis=2)
        elif depth_image.ndim == 3 and depth_image.shape[2] != 1:
            depth_image = depth_image[:, :, 0:1]

        # 确保 RGB 和 Depth 尺寸一致
        if rgb_image.shape[:2] != depth_image.shape[:2]:
            depth_image = cv2.resize(
                depth_image,
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if depth_image.ndim == 2:
                depth_image = np.expand_dims(depth_image, axis=2)

    # 对于纯rgb或depth输入，可以进行相应处理，这里主要关注rgbd
    elif input_type == "rgb":
        # 创建一个全零的深度图作为占位符
        depth_image = np.zeros(
            (rgb_image.shape[0], rgb_image.shape[1], 1), dtype=np.float32
        )

    return rgb_image, depth_image


def main(args):
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cfg = setup_cfg(args.config, args.weights)
    predictor = RGBDPredictor(cfg)

    # 加载分离的 RGB 和 Depth 图像
    rgb_img, depth_img = load_image_depth(args.input, args.input_type)

    if args.input_type == "rgbd" and depth_img is None:
        print("错误：选择了 rgbd 输入类型但无法加载深度图像。")
        return

    print("正在进行预测...")
    outputs = predictor(rgb_img, depth_img)
    print("预测完成。")

    instances = outputs["instances"].to("cpu")
    # --- 可选的改进：根据置信度分数进行筛选 ---
    confidence_threshold = 0.7  # 设置一个置信度阈值
    high_conf_indices = instances.scores > confidence_threshold

    scores = instances.scores[high_conf_indices]
    masks = instances.pred_masks[high_conf_indices].numpy()
    # -----------------------------------------
    # --- 开始修改：生成彩色可视化结果 ---

    # 创建一个鲜艳的颜色调色板 (R, G, B格式)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (192, 192, 192),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (255, 165, 0),
        (255, 20, 147),
    ]

    # 创建一个原始图像的副本，用于绘制彩色掩码
    overlay = rgb_img.copy()
    alpha = 0.5  # 设置掩码的透明度

    print(f"检测到 {len(masks)} 个实例，正在生成彩色掩码...")
    for i, mask in enumerate(masks):
        # 将浮点数mask转换为布尔型mask，阈值为0.5
        boolean_mask = mask > 0.5

        color = colors[i % len(colors)]
        # 使用布尔型mask作为索引
        overlay[boolean_mask] = (
            overlay[boolean_mask] * (1 - alpha)
            + np.array(color, dtype=np.uint8) * alpha
        )

    # 将最终的 RGB 图像转换为 BGR 以便 cv2 保存
    output_image_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    out_path = args.output if args.output else "output.png"
    cv2.imwrite(out_path, output_image_bgr)
    print(f"彩色分割结果已保存到: {out_path}")
    # --- 结束修改 ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MGM RGB-D 目标分割推理脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--weights", required=True, help="权重文件路径")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument(
        "--input-type", choices=["rgb", "rgbd"], default="rgbd", help="输入类型"
    )
    parser.add_argument("--output", help="输出结果路径")
    args = parser.parse_args()
    main(args)

    """
    使用示例：
python predict_mgm.py \
    --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml \
    --weights MGM_Mask2Former/pretrained-checkpoint/0909_2K_REAL_0909_5K.pth  \
    --input MGM_Mask2Former/predict_test/3521_7843763521_25_scene_000001_000001_v0.png  \
    --input-type rgbd  \
    --output /home/fuyx/zhn/mask2former/MGM_Mask2Former/predict_test/output/3521_7843763521_25_scene_000001_000001_v0_segment_result.png
    
    python3 MGM_Mask2Former/predict_mgm.py --config MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml --weights output/0909_0.12K/0909_10K/model_final.pth --input MGM_Mask2Former/predict_test/512/3521_7843763521_25_scene_000001_000001_v0.png --input-type rgbd --output MGM_Mask2Former/predict_test/512/output/3521_7843763521_25_scene_000001_000001_v0_segment_result.png
    """
