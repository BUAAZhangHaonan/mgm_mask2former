# -*- coding: utf-8 -*-
import torch
from typing import Dict, Tuple
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg

from mask2former import COCOInstanceRGBDDatasetMapper, build_mgm


@META_ARCH_REGISTRY.register()
class BackbonesToMGM(Backbone):
    def __init__(self, cfg):
        super().__init__()
        rgb_cfg = cfg.clone()
        rgb_cfg.defrost()
        rgb_cfg.MODEL.BACKBONE.NAME = cfg.MODEL.RGB_BACKBONE.NAME
        rgb_cfg.MODEL.SWIN = cfg.MODEL.RGB_BACKBONE.SWIN
        rgb_cfg.freeze()
        self.rgb_backbone = build_backbone(rgb_cfg)

        depth_cfg = cfg.clone()
        depth_cfg.defrost()
        depth_cfg.MODEL.BACKBONE.NAME = cfg.MODEL.DEPTH_BACKBONE.NAME
        depth_cfg.freeze()
        self.depth_backbone = build_backbone(depth_cfg)

        self.mgm = build_mgm(cfg)
        self._out_features = list(self.rgb_backbone.output_shape().keys())
        self._out_feature_strides = {
            k: v.stride for k, v in self.rgb_backbone.output_shape().items()
        }
        self._out_feature_channels = {
            k: v.channels for k, v in self.rgb_backbone.output_shape().items()
        }

    def forward(
        self, batched_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        rgb_images = batched_inputs["image"]
        depth_images = batched_inputs.get("depth", batched_inputs.get("depth_image"))
        depth_raw = batched_inputs.get("depth_raw", depth_images)

        rgb_features = self.rgb_backbone(rgb_images)
        depth_features = self.depth_backbone(depth_images)

        fused_features, confidence_maps, _ = self.mgm(
            image_features=rgb_features,
            depth_features=depth_features,
            depth_raw=depth_raw,
            rgb_image=rgb_images,
        )
        return fused_features, confidence_maps

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


class Demo:
    def __init__(self, args):
        self.args = args
        self.cfg = self._setup_cfg(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BackbonesToMGM(self.cfg).to(self.device).eval()
        self.pixel_mean = torch.tensor(
            self.cfg.MODEL.PIXEL_MEAN, device=self.device
        ).view(3, 1, 1)
        self.pixel_std = torch.tensor(
            self.cfg.MODEL.PIXEL_STD, device=self.device
        ).view(3, 1, 1)
        print(f"Demo initialized successfully. Using device: {self.device}")

    def _setup_cfg(self, args):
        cfg = get_cfg()
        from mask2former.modeling.config.mgm_config import add_mgm_config

        add_mgm_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        return cfg

    def run(self):
        if self.args.mode == "random":
            self.run_with_random_data()
        elif self.args.mode == "real":
            self.run_with_real_data()
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")

    def run_with_random_data(self):
        print("\n--- Running test with random data (no mapper) ---")
        random_rgb_image = torch.randint(0, 256, (1, 3, 512, 640), dtype=torch.uint8)
        random_depth_image = torch.rand(1, 1, 512, 640, dtype=torch.float32)
        rgb_tensor = (
            random_rgb_image.float().to(self.device) - self.pixel_mean
        ) / self.pixel_std
        batched_inputs = {
            "image": rgb_tensor,
            "depth_image": random_depth_image.to(self.device),
            "depth_raw": random_depth_image.to(self.device),
        }
        print(f"Input image shape: {batched_inputs['image'].shape}")
        with torch.no_grad():
            fused_features, confidence_maps = self.model(batched_inputs)
        print("Forward pass successful!")
        self._print_and_save_results(fused_features, confidence_maps, "random_test")
        print("--- Random data test finished ---")

    def run_with_real_data(self):
        print("\n--- Running test with real data (using mapper) ---")
        mapper = COCOInstanceRGBDDatasetMapper(self.cfg, is_train=False)
        dataset_root = self.cfg.INPUT.DATASET_ROOT
        split = self.args.split
        image_root = os.path.join(dataset_root, "images", split)
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        image_files = sorted(
            [
                f
                for f in os.listdir(image_root)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if not image_files:
            print(f"No images found in {image_root}. Exiting.")
            return

        for image_file in image_files:
            print(f"Processing: {image_file}")
            image_path = os.path.join(image_root, image_file)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                dataset_dict = {
                    "file_name": image_path,
                    "height": height,
                    "width": width,
                }
                data_dict = mapper(dataset_dict)
            except Exception as e:
                print(f"  Skipping {image_file} due to mapper error: {e}")
                continue

            image_tensor_float = (
                data_dict["image"].to(self.device).float() - self.pixel_mean
            ) / self.pixel_std
            depth_tensor = data_dict["depth"].to(self.device)
            batched_inputs = {
                "image": image_tensor_float.unsqueeze(0),
                "depth": depth_tensor.unsqueeze(0),
                "depth_raw": depth_tensor.unsqueeze(0),
            }
            with torch.no_grad():
                fused_features, confidence_maps = self.model(batched_inputs)
            print("  Forward pass successful!")
            self._print_and_save_results(
                fused_features, confidence_maps, os.path.splitext(image_file)[0]
            )
        print("--- Real data test finished ---")

    def _print_and_save_results(self, fused_features, confidence_maps, prefix):
        print("  Fused features shapes:")
        for k, v in fused_features.items():
            print(f"    {k}: {tuple(v.shape)}")
        print("  Confidence maps shapes:")
        for k, v in confidence_maps.items():
            print(f"    {k}: {tuple(v.shape)}")
        self._save_confidence_maps(confidence_maps, self.args.output, prefix)
        print(f"  Saved confidence maps to {self.args.output}")

    def _save_confidence_maps(
        self, conf_maps: Dict[str, torch.Tensor], output_dir: str, prefix: str
    ):
        os.makedirs(output_dir, exist_ok=True)
        for scale, tensor in conf_maps.items():
            # Extract single image from batch, move to CPU, and convert to numpy
            img_tensor = tensor[0, 0].detach().cpu()
            img_arr_float = img_tensor.numpy()

            # Choose visualization mode based on --colormap argument
            if self.args.colormap:
                try:
                    cmap = plt.get_cmap(self.args.colormap)
                    # Apply colormap (returns RGBA), convert to RGB, and scale to 0-255
                    colored_arr = (cmap(img_arr_float)[:, :, :3] * 255).astype(np.uint8)
                    Image.fromarray(colored_arr).save(
                        os.path.join(output_dir, f"{prefix}_conf_{scale}.png")
                    )
                except ValueError:
                    print(
                        f"Warning: Colormap '{self.args.colormap}' not found. Falling back to grayscale."
                    )
                    # Fallback to grayscale if colormap name is invalid
                    gray_arr = (img_arr_float * 255.0).astype("uint8")
                    Image.fromarray(gray_arr).save(
                        os.path.join(output_dir, f"{prefix}_conf_{scale}.png")
                    )
            else:
                # Original grayscale saving
                gray_arr = (img_arr_float * 255.0).astype("uint8")
                Image.fromarray(gray_arr).save(
                    os.path.join(output_dir, f"{prefix}_conf_{scale}_gray.png")
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MGM Full Pipeline Test Runner")
    parser.add_argument("--config-file", required=True, help="Path to the config file")
    parser.add_argument(
        "--mode", required=True, choices=["random", "real"], help="Test mode"
    )
    parser.add_argument(
        "--split", default="test", help="[For 'real' mode] The dataset split to process"
    )
    parser.add_argument(
        "--output",
        default="MGM_Mask2Former/mgm_test_output_random",
        help="Directory to save the output maps",
    )

    parser.add_argument(
        "--colormap",
        type=str,
        default=None,
        help="Apply a matplotlib colormap (e.g., 'jet', 'viridis', 'plasma') to the confidence map for better visualization.",
    )

    parser.add_argument(
        "opts", default=[], nargs=argparse.REMAINDER, help="Modify config options"
    )
    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
