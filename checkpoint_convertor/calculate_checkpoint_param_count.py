import torch
import os
from collections import defaultdict


def load_checkpoint_safely(checkpoint_path):
    """安全加载checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: File does not exist: {checkpoint_path}")
        return None

    file_size = os.path.getsize(checkpoint_path)
    print(f"File size: {file_size / (1024*1024):.2f} MB")

    # 尝试多种加载方式
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=True)
        print("Successfully loaded with weights_only=True")
    except Exception as e:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)
            print("Successfully loaded with weights_only=False")
        except Exception as e2:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print("Successfully loaded without weights_only parameter")
            except Exception as e3:
                print(f"Failed to load checkpoint: {e3}")
                return None

    return checkpoint


def get_state_dict_from_checkpoint(checkpoint):
    """从checkpoint中提取state_dict"""
    possible_keys = ['model', 'state_dict', 'model_state_dict']

    for key in possible_keys:
        if key in checkpoint:
            print(f"Using '{key}' as state_dict")
            return checkpoint[key]

    # 如果没有找到，可能整个checkpoint就是state_dict
    print("Using checkpoint directly as state_dict")
    return checkpoint


def calculate_param_count(state_dict):
    """计算参数数量统计"""
    # 按模块分组统计
    module_stats = defaultdict(
        lambda: {'params': 0, 'tensors': 0, 'details': []})
    total_params = 0

    print(f"\n=== Parameter Analysis ===")
    print(f"Total tensors in state_dict: {len(state_dict)}")

    for param_name, param_tensor in state_dict.items():
        if torch.is_tensor(param_tensor):
            param_count = param_tensor.numel()
            total_params += param_count

            # 提取主模块名
            parts = param_name.split('.')
            if len(parts) >= 2:
                main_module = parts[0]
                if len(parts) >= 3 and parts[0] == 'sem_seg_head':
                    # 对于sem_seg_head，进一步细分
                    main_module = f"{parts[0]}.{parts[1]}"
            else:
                main_module = parts[0]

            module_stats[main_module]['params'] += param_count
            module_stats[main_module]['tensors'] += 1
            module_stats[main_module]['details'].append({
                'name': param_name,
                'shape': list(param_tensor.shape),
                'params': param_count
            })

    return module_stats, total_params


def print_detailed_stats(module_stats, total_params):
    """打印详细统计信息"""
    print(f"\n=== Module Parameter Statistics ===")
    print(f"{'Module':<40} {'Parameters':<15} {'Tensors':<10} {'Percentage':<12}")
    print("-" * 80)

    # 按参数数量排序
    sorted_modules = sorted(module_stats.items(),
                            key=lambda x: x[1]['params'], reverse=True)

    for module_name, stats in sorted_modules:
        percentage = (stats['params'] / total_params) * 100
        print(
            f"{module_name:<40} {stats['params']:>14,} {stats['tensors']:>9} {percentage:>10.2f}%")

    print("-" * 80)
    print(
        f"{'TOTAL':<40} {total_params:>14,} {sum(s['tensors'] for s in module_stats.values()):>9} {100.0:>10.2f}%")


def print_module_details(module_stats, show_details=False):
    """打印每个模块的详细信息"""
    if not show_details:
        return

    print(f"\n=== Detailed Parameter Breakdown ===")
    for module_name, stats in sorted(module_stats.items()):
        print(f"\n--- {module_name} ---")
        print(f"Total parameters: {stats['params']:,}")
        print(f"Total tensors: {stats['tensors']}")

        # 显示前10个最大的参数
        sorted_details = sorted(
            stats['details'], key=lambda x: x['params'], reverse=True)
        print("Top parameters:")
        for i, detail in enumerate(sorted_details[:10]):
            print(
                f"  {detail['name']}: {detail['shape']} ({detail['params']:,} params)")

        if len(sorted_details) > 10:
            print(f"  ... and {len(sorted_details) - 10} more parameters")


def analyze_checkpoint_structure(checkpoint_path, show_details=False):
    """主函数：分析checkpoint结构和参数"""
    # 加载checkpoint
    checkpoint = load_checkpoint_safely(checkpoint_path)
    if checkpoint is None:
        return

    # 获取state_dict
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    if state_dict is None:
        print("Failed to extract state_dict from checkpoint")
        return

    # 计算参数统计
    module_stats, total_params = calculate_param_count(state_dict)

    # 打印统计结果
    print_detailed_stats(module_stats, total_params)

    # 打印详细信息（可选）
    print_module_details(module_stats, show_details)

    return module_stats, total_params


if __name__ == "__main__":
    # 配置要分析的checkpoint路径
    checkpoint_paths = [
        "/home/remote1/zhanghaonan/projects/mask2former_0822/test/convnext-checkpoint/mask2former_coco_swin_t.pth",
        # 可以添加更多路径进行对比
    ]

    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*60}")
            print(f"Analyzing: {os.path.basename(checkpoint_path)}")
            print(f"{'='*60}")

            try:
                analyze_checkpoint_structure(
                    checkpoint_path, show_details=True)
            except Exception as e:
                print(f"Error analyzing {checkpoint_path}: {e}")
        else:
            print(f"File not found: {checkpoint_path}")
