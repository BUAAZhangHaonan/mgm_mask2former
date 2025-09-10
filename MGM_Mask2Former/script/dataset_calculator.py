import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def calculate_rgb_stats_for_dir(rgb_dir):
    """
    计算单个目录中所有 RGB 图像的均值和标准差。
    """
    print(f"正在处理目录: {rgb_dir}")

    try:
        image_files = [
            f for f in os.listdir(rgb_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        if not image_files:
            print(f"警告: 在 '{rgb_dir}' 中没有找到图像文件。")
            return None, None
    except FileNotFoundError:
        print(f"错误: 目录 '{rgb_dir}' 不存在。")
        return None, None

    # 初始化用于计算的变量
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for filename in tqdm(image_files, desc=f"计算 {os.path.basename(rgb_dir)}"):
        rgb_path = os.path.join(rgb_dir, filename)
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_image is None:
            print(f"\n警告: 无法读取 RGB 图像 {rgb_path}, 已跳过。")
            continue

        # 将 BGR 转换为 RGB
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        h, w, _ = rgb_image.shape
        pixel_count += h * w

        rgb_image_float = rgb_image.astype(np.float64)
        rgb_sum += np.sum(rgb_image_float, axis=(0, 1))
        rgb_sum_sq += np.sum(np.square(rgb_image_float), axis=(0, 1))

    if pixel_count == 0:
        print(f"错误: 在 '{rgb_dir}' 中没有成功处理任何像素。")
        return None, None

    # 计算最终的均值和标准差
    rgb_mean = rgb_sum / pixel_count
    rgb_std = np.sqrt(rgb_sum_sq / pixel_count - np.square(rgb_mean))

    return rgb_mean, rgb_std


def main(args):
    """
    遍历数据集根目录下的 train, test, val 子集并计算统计信息。
    """
    dataset_root = args.dataset_root
    subsets = ["train", "test", "val"]

    print(f"开始计算数据集根目录 '{dataset_root}' 的统计信息...")
    print("=" * 60)

    for subset in subsets:
        print(f"\n--- 开始处理 '{subset}' 子数据集 ---")

        # 根据文件存储规则构建图像目录路径
        rgb_dir = os.path.join(dataset_root, "images", subset)

        mean, std = calculate_rgb_stats_for_dir(rgb_dir)

        if mean is not None and std is not None:
            print(f"\n--- '{subset}' 子数据集计算结果 ---")
            print("可将以下参数用于该子集的配置：")
            print("MODEL:")
            print(
                "  PIXEL_MEAN: [{:.4f}, {:.4f}, {:.4f}]".format(
                    mean[0], mean[1], mean[2]
                )
            )
            print(
                "  PIXEL_STD: [{:.4f}, {:.4f}, {:.4f}]".format(std[0], std[1], std[2])
            )
            print("-" * 40)

    print("\n" + "=" * 60)
    print("所有子集处理完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="计算数据集中 train/test/val 子集的 RGB 均值和标准差"
    )
    # 输入参数变为数据集根目录
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/fuyx/zhn/mgm_datasets/dataset_0909_512_LESS",
        help="数据集的根目录 (例如: mgm_test_input)",
    )
    args = parser.parse_args()

    main(args)
