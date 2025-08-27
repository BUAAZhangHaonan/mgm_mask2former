#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集验证综合工具
功能：数据集注册 + 完整性检查 + 可视化验证

输入：数据集配置参数
输出：验证报告 + 可视化样本 + 训练配置建议

使用方法: python validate_dataset_tool.py --dataset-name my_dataset --data-root datasets/my_dataset

这个工具整合了原来的三个脚本功能：
1. 数据集注册和验证（原register_custom_dataset.py）
2. 数据集完整性检查和统计分析（原visualize_dataset.py）  
3. 生成训练配置建议（原generate_config.py的核心功能）
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional

# Detectron2 imports
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.utils.visualizer import Visualizer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mask2former.data.datasets.register_coco_rgbd_instance import register_coco_rgbd_instances


class DatasetValidator:
    """数据集验证器 - 整合所有验证功能"""
    
    def __init__(self, dataset_name: str, data_root: str, categories: List[Dict]):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.categories = categories
        self.validation_results = {}
        
    def get_metadata(self):
        """获取数据集元数据"""
        thing_ids = [cat["id"] for cat in self.categories]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [cat["name"] for cat in self.categories]
        
        return {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
        }
    
    def check_dataset_structure(self) -> bool:
        """检查数据集目录结构"""
        print("🔍 检查数据集结构...")
        
        required_structure = {
            "images/train": "训练图像目录",
            "images/val": "验证图像目录", 
            "annotations/instances_train.json": "训练标注文件",
            "annotations/instances_val.json": "验证标注文件"
        }
        
        optional_structure = {
            "depth/train": "训练深度图像目录",
            "depth/val": "验证深度图像目录"
        }
        
        structure_ok = True
        has_depth = True
        
        # 检查必需文件
        for path, desc in required_structure.items():
            full_path = self.data_root / path
            if full_path.exists():
                print(f"  ✅ {desc}: {full_path}")
            else:
                print(f"  ❌ {desc}: {full_path} (缺失)")
                structure_ok = False
        
        # 检查可选文件（深度数据）
        for path, desc in optional_structure.items():
            full_path = self.data_root / path
            if full_path.exists():
                print(f"  ✅ {desc}: {full_path}")
            else:
                print(f"  ⚠️  {desc}: {full_path} (可选，RGB-D模式需要)")
                has_depth = False
        
        self.validation_results['structure_ok'] = structure_ok
        self.validation_results['has_depth'] = has_depth
        return structure_ok
    
    def validate_annotations(self) -> bool:
        """验证标注文件格式"""
        print("\n📋 验证标注文件...")
        
        annotation_files = {
            "train": self.data_root / "annotations/instances_train.json",
            "val": self.data_root / "annotations/instances_val.json"
        }
        
        annotations_ok = True
        dataset_stats = {}
        
        for split, ann_file in annotation_files.items():
            if not ann_file.exists():
                continue
                
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查必需字段
                required_fields = ['images', 'annotations', 'categories']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"  ❌ {split}: 缺少字段 {missing_fields}")
                    annotations_ok = False
                    continue
                
                # 统计信息
                num_images = len(data['images'])
                num_annotations = len(data['annotations'])
                num_categories = len(data['categories'])
                
                print(f"  ✅ {split}: {num_images} 图像, {num_annotations} 标注, {num_categories} 类别")
                
                dataset_stats[split] = {
                    'num_images': num_images,
                    'num_annotations': num_annotations,
                    'num_categories': num_categories,
                    'avg_annotations_per_image': num_annotations / num_images if num_images > 0 else 0
                }
                
                # 检查图像文件是否存在
                image_dir = self.data_root / f"images/{split}"
                missing_images = 0
                for img_info in data['images'][:10]:  # 只检查前10个
                    img_path = image_dir / img_info['file_name']
                    if not img_path.exists():
                        missing_images += 1
                
                if missing_images > 0:
                    print(f"    ⚠️  发现 {missing_images}/10 个图像文件缺失（抽样检查）")
                
            except Exception as e:
                print(f"  ❌ {split}: 标注文件解析错误 - {e}")
                annotations_ok = False
        
        self.validation_results['annotations_ok'] = annotations_ok
        self.validation_results['dataset_stats'] = dataset_stats
        return annotations_ok
    
    def register_datasets(self) -> bool:
        """注册数据集"""
        print("\n📝 注册数据集...")
        
        metadata = self.get_metadata()
        registration_ok = True
        
        try:
            # 注册RGB数据集
            for split in ['train', 'val']:
                dataset_name = f"{self.dataset_name}_{split}"
                ann_file = self.data_root / f"annotations/instances_{split}.json"
                img_dir = self.data_root / f"images/{split}"
                
                if ann_file.exists() and img_dir.exists():
                    register_coco_instances(
                        dataset_name,
                        metadata,
                        str(ann_file),
                        str(img_dir)
                    )
                    print(f"  ✅ 已注册RGB数据集: {dataset_name}")
            
            # 如果有深度数据，注册RGB-D数据集
            if self.validation_results.get('has_depth', False):
                for split in ['train', 'val']:
                    dataset_name = f"{self.dataset_name}_rgbd_{split}"
                    ann_file = self.data_root / f"annotations/instances_{split}.json"
                    img_dir = self.data_root / f"images/{split}"
                    depth_dir = self.data_root / f"depth/{split}"
                    
                    if all(p.exists() for p in [ann_file, img_dir, depth_dir]):
                        register_coco_rgbd_instances(
                            dataset_name,
                            metadata,
                            str(ann_file),
                            str(img_dir),
                            str(depth_dir)
                        )
                        print(f"  ✅ 已注册RGB-D数据集: {dataset_name}")
            
        except Exception as e:
            print(f"  ❌ 数据集注册失败: {e}")
            registration_ok = False
        
        self.validation_results['registration_ok'] = registration_ok
        return registration_ok
    
    def analyze_dataset_statistics(self):
        """分析数据集统计信息"""
        print("\n📊 数据集统计分析...")
        
        for split in ['train', 'val']:
            dataset_name = f"{self.dataset_name}_{split}"
            
            try:
                dataset_dicts = DatasetCatalog.get(dataset_name)
                metadata = MetadataCatalog.get(dataset_name)
                
                print(f"\n{split.upper()}集统计:")
                print(f"  - 图像总数: {len(dataset_dicts)}")
                
                # 类别统计
                category_counts = {}
                image_sizes = []
                annotation_areas = []
                
                for d in dataset_dicts:
                    # 图像尺寸
                    if "height" in d and "width" in d:
                        image_sizes.append((d["width"], d["height"]))
                    
                    # 标注统计
                    if "annotations" in d:
                        for ann in d["annotations"]:
                            cat_id = ann["category_id"]
                            if cat_id in metadata.thing_dataset_id_to_contiguous_id:
                                contiguous_id = metadata.thing_dataset_id_to_contiguous_id[cat_id]
                                cat_name = metadata.thing_classes[contiguous_id]
                                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                            
                            if "area" in ann:
                                annotation_areas.append(ann["area"])
                
                # 显示类别分布
                total_annotations = sum(category_counts.values())
                print(f"  - 标注总数: {total_annotations}")
                
                for cat_name, count in sorted(category_counts.items()):
                    percentage = count / total_annotations * 100 if total_annotations > 0 else 0
                    print(f"    {cat_name}: {count} ({percentage:.1f}%)")
                
                # 图像尺寸统计
                if image_sizes:
                    widths = [size[0] for size in image_sizes]
                    heights = [size[1] for size in image_sizes]
                    print(f"  - 图像尺寸范围: {min(widths)}x{min(heights)} ~ {max(widths)}x{max(heights)}")
                
            except Exception as e:
                print(f"  ❌ {split}集分析失败: {e}")
    
    def visualize_samples(self, num_samples: int = 3, save_dir: Optional[str] = None):
        """可视化数据集样本"""
        print(f"\n🎨 可视化数据集样本...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for split in ['train', 'val']:
            dataset_name = f"{self.dataset_name}_{split}"
            
            try:
                dataset_dicts = DatasetCatalog.get(dataset_name)
                metadata = MetadataCatalog.get(dataset_name)
                
                if not dataset_dicts:
                    continue
                
                samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
                
                for i, d in enumerate(samples):
                    # 读取RGB图像
                    img = cv2.imread(d["file_name"])
                    if img is None:
                        continue
                    
                    # 可视化标注
                    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
                    if "annotations" in d:
                        vis = visualizer.draw_dataset_dict(d)
                    else:
                        vis = visualizer.get_output()
                    
                    result_img = vis.get_image()[:, :, ::-1]
                    
                    # 检查是否有深度图像
                    has_depth_sample = False
                    if f"{self.dataset_name}_rgbd_{split}" in DatasetCatalog.list():
                        rgbd_dataset_dicts = DatasetCatalog.get(f"{self.dataset_name}_rgbd_{split}")
                        for rgbd_d in rgbd_dataset_dicts:
                            if rgbd_d["file_name"] == d["file_name"] and "depth_file_name" in rgbd_d:
                                depth_img = cv2.imread(rgbd_d["depth_file_name"], cv2.IMREAD_UNCHANGED)
                                if depth_img is not None:
                                    # 深度图可视化
                                    depth_normalized = np.zeros_like(depth_img, dtype=np.uint8)
                                    cv2.normalize(depth_img, depth_normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                                    
                                    # 组合RGB和深度图
                                    h, w = img.shape[:2]
                                    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
                                    combined[:, :w] = result_img
                                    combined[:, w:] = depth_colored
                                    result_img = combined
                                    has_depth_sample = True
                                break
                    
                    # 保存或显示
                    title = f"{split}_sample_{i+1}" + ("_rgbd" if has_depth_sample else "_rgb")
                    
                    if save_dir:
                        save_path = os.path.join(save_dir, f"{title}.jpg")
                        cv2.imwrite(save_path, result_img)
                        print(f"  💾 已保存: {save_path}")
                    else:
                        cv2.imshow(title, result_img)
                        
                if not save_dir:
                    print("  👁️  按任意键查看下一组样本...")
                    cv2.waitKey(0)
                    
            except Exception as e:
                print(f"  ❌ {split}集可视化失败: {e}")
        
        if not save_dir:
            cv2.destroyAllWindows()
    
    def generate_training_suggestions(self) -> Dict:
        """生成训练配置建议"""
        print("\n💡 生成训练配置建议...")
        
        stats = self.validation_results.get('dataset_stats', {})
        has_depth = self.validation_results.get('has_depth', False)
        
        suggestions = {
            'dataset_info': {
                'name': self.dataset_name,
                'num_classes': len(self.categories),
                'has_depth_data': has_depth,
                'recommended_modes': ['RGB'] + (['RGB-D'] if has_depth else [])
            },
            'training_config': {}
        }
        
        # 根据数据集大小推荐训练参数
        train_stats = stats.get('train', {})
        if train_stats:
            num_images = train_stats['num_images']
            
            # 推荐batch size
            if has_depth:
                batch_size = 4 if num_images > 1000 else 2
            else:
                batch_size = 8 if num_images > 1000 else 4
            
            # 推荐训练迭代数
            epochs = 50 if num_images < 5000 else 30
            max_iter = max(1000, (num_images // batch_size) * epochs)
            
            suggestions['training_config'] = {
                'batch_size_rgb': 8 if num_images > 1000 else 4,
                'batch_size_rgbd': 4 if num_images > 1000 else 2,
                'max_iter': max_iter,
                'learning_rate': 0.0001,
                'image_size': 1024 if num_images > 2000 else 512
            }
        
        # 打印建议
        print(f"  📋 数据集信息:")
        print(f"    - 数据集名称: {suggestions['dataset_info']['name']}")
        print(f"    - 类别数量: {suggestions['dataset_info']['num_classes']}")
        print(f"    - 支持模式: {', '.join(suggestions['dataset_info']['recommended_modes'])}")
        
        if suggestions['training_config']:
            print(f"  ⚙️  训练配置建议:")
            config = suggestions['training_config']
            print(f"    - RGB批次大小: {config['batch_size_rgb']}")
            if has_depth:
                print(f"    - RGB-D批次大小: {config['batch_size_rgbd']}")
            print(f"    - 最大迭代数: {config['max_iter']}")
            print(f"    - 学习率: {config['learning_rate']}")
            print(f"    - 图像大小: {config['image_size']}")
        
        return suggestions
    
    def print_usage_instructions(self, suggestions: Dict):
        """打印使用说明"""
        print(f"\n📖 使用说明:")
        print(f"现在你可以在train_net.py中添加以下注册代码：")
        print(f"")
        print(f"# 在train_net.py开头添加数据集注册")
        print(f"def register_custom_dataset():")
        print(f"    from detectron2.data.datasets.coco import register_coco_instances")
        print(f"    metadata = {{")
        print(f"        'thing_dataset_id_to_contiguous_id': {self.get_metadata()['thing_dataset_id_to_contiguous_id']},")
        print(f"        'thing_classes': {self.get_metadata()['thing_classes']}")
        print(f"    }}")
        print(f"    ")
        print(f"    # RGB数据集")
        print(f"    register_coco_instances('{self.dataset_name}_train', metadata, '{self.data_root}/annotations/instances_train.json', '{self.data_root}/images/train')")
        print(f"    register_coco_instances('{self.dataset_name}_val', metadata, '{self.data_root}/annotations/instances_val.json', '{self.data_root}/images/val')")
        
        if suggestions['dataset_info']['has_depth_data']:
            print(f"    ")
            print(f"    # RGB-D数据集")
            print(f"    from mask2former.data.datasets.register_coco_rgbd_instance import register_coco_rgbd_instances")
            print(f"    register_coco_rgbd_instances('{self.dataset_name}_rgbd_train', metadata, '{self.data_root}/annotations/instances_train.json', '{self.data_root}/images/train', '{self.data_root}/depth/train')")
            print(f"    register_coco_rgbd_instances('{self.dataset_name}_rgbd_val', metadata, '{self.data_root}/annotations/instances_val.json', '{self.data_root}/images/val', '{self.data_root}/depth/val')")
        
        print(f"")
        print(f"register_custom_dataset()  # 在main()函数之前调用")
        print(f"")
        print(f"配置文件中设置：")
        print(f"DATASETS:")
        print(f"  TRAIN: ('{self.dataset_name}_train',)")
        print(f"  TEST: ('{self.dataset_name}_val',)")
        print(f"MODEL:")
        print(f"  SEM_SEG_HEAD:")
        print(f"    NUM_CLASSES: {len(self.categories)}")
        
        if suggestions['training_config']:
            config = suggestions['training_config']
            print(f"SOLVER:")
            print(f"  IMS_PER_BATCH: {config['batch_size_rgb']}")
            print(f"  MAX_ITER: {config['max_iter']}")
            print(f"INPUT:")
            print(f"  IMAGE_SIZE: {config['image_size']}")
            print(f"  DATASET_MAPPER_NAME: 'coco_instance_lsj'  # RGB模式")
    
    def run_validation(self, visualize: bool = True, save_samples: Optional[str] = None) -> bool:
        """运行完整验证流程"""
        print(f"🚀 开始验证数据集: {self.dataset_name}")
        print(f"📁 数据根目录: {self.data_root}")
        print(f"=" * 60)
        
        # 步骤1: 检查目录结构
        if not self.check_dataset_structure():
            print("❌ 数据集结构检查失败")
            return False
        
        # 步骤2: 验证标注文件
        if not self.validate_annotations():
            print("❌ 标注文件验证失败")
            return False
        
        # 步骤3: 注册数据集
        if not self.register_datasets():
            print("❌ 数据集注册失败")
            return False
        
        # 步骤4: 统计分析
        self.analyze_dataset_statistics()
        
        # 步骤5: 可视化验证
        if visualize:
            self.visualize_samples(save_dir=save_samples)
        
        # 步骤6: 生成建议
        suggestions = self.generate_training_suggestions()
        
        # 步骤7: 打印使用说明
        self.print_usage_instructions(suggestions)
        
        print(f"\n✅ 数据集验证完成！数据集可用于训练。")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="数据集验证综合工具",
        epilog="""
使用示例:
  python validate_dataset_tool.py --dataset-name my_dataset --data-root datasets/my_dataset --categories person,vehicle
  python validate_dataset_tool.py --dataset-name custom_data --data-root /path/to/data --config categories.json --save-samples output/samples
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dataset-name", required=True, help="数据集名称")
    parser.add_argument("--data-root", required=True, help="数据集根目录路径")
    parser.add_argument("--categories", help="类别列表，用逗号分隔，如: person,vehicle,object")
    parser.add_argument("--config", help="类别配置JSON文件路径")
    parser.add_argument("--no-visualize", action="store_true", help="跳过可视化步骤")
    parser.add_argument("--save-samples", help="保存可视化样本的目录（如果不指定则显示）")
    
    args = parser.parse_args()
    
    # 解析类别信息
    categories = []
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            categories = json.load(f)
    elif args.categories:
        cat_names = args.categories.split(',')
        categories = [
            {"id": i+1, "name": name.strip(), "supercategory": name.strip()}
            for i, name in enumerate(cat_names)
        ]
    else:
        print("❌ 请提供类别信息（--categories 或 --config）")
        return False
    
    # 创建验证器并运行
    validator = DatasetValidator(args.dataset_name, args.data_root, categories)
    success = validator.run_validation(
        visualize=not args.no_visualize,
        save_samples=args.save_samples
    )
    
    if success:
        print("✅ 数据集验证成功")
    else:
        print("❌ 数据集验证失败")


if __name__ == "__main__":
    main()
