# demo/bench_mask2former.py
import argparse
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from tqdm import tqdm


def setup_cfg(config_file, weights, conf_threshold=0.5, device="cuda"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD.TEST_SCORE_THRESH = conf_threshold
    cfg.freeze()
    return cfg


def autocast_ctx_for(device, amp_enabled):
    if device == "cuda":
        return torch.amp.autocast("cuda", enabled=amp_enabled)
    # CPU 默认不启用 AMP（bfloat16 autocast 只在部分 CPU 上有效，且收益有限）
    return torch.amp.autocast("cpu", enabled=False)


def run_predictor(predictor, image_bgr, iters=100, warmup=10, device="cuda", amp_enabled=False):
    use_cuda = device == "cuda" and torch.cuda.is_available()

    # 预热
    with torch.inference_mode(), autocast_ctx_for(device, amp_enabled):
        for _ in tqdm(range(warmup), desc=f"[{device}] Warmup", leave=False):
            _ = predictor(image_bgr)
    if use_cuda:
        torch.cuda.synchronize()

    # 计时
    t0 = time.time()
    with torch.inference_mode(), autocast_ctx_for(device, amp_enabled):
        for _ in tqdm(range(iters), desc=f"[{device}] Benchmark", leave=False):
            _ = predictor(image_bgr)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / iters


def sanity_check(predictor, image_bgr, device, amp_enabled):
    with torch.inference_mode(), autocast_ctx_for(device, amp_enabled):
        pred = predictor(image_bgr)
    if isinstance(pred, dict) and "instances" in pred and pred["instances"] is not None:
        try:
            n_inst = len(pred["instances"])
        except Exception:
            n_inst = "N/A"
        print(f"[{device}] Sanity check: instances = {n_inst}")
    else:
        keys = list(pred.keys()) if isinstance(pred, dict) else type(pred)
        print(f"[{device}] Sanity check: output keys = {keys}")


def main():
    parser = argparse.ArgumentParser(description="Mask2Former benchmark (GPU AMP and CPU)")
    parser.add_argument("--config-file", required=True, help="path to config (.yaml)")
    parser.add_argument("--weights", required=True, help="path to weights (.pkl or .pth)")
    parser.add_argument("--input", required=True, help="path to a single input image")
    parser.add_argument("--iters", type=int, default=100, help="timing iterations")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--amp", choices=["off", "on", "both"], default="both",
                        help="GPU autocast mode to benchmark")
    parser.add_argument("--device", choices=["cuda", "cpu", "both"], default="both",
                        help="which device(s) to benchmark")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="set torch.set_num_threads for CPU runs")
    args = parser.parse_args()

    assert os.path.isfile(args.input), f"Input not found: {args.input}"
    assert os.path.isfile(args.weights), f"Weights not found: {args.weights}"

    # 固定输入尺寸时可开启 cuDNN benchmark；仅在 CUDA 跑时生效
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 读取图像（BGR）
    image = read_image(args.input, format="BGR")

    devices = []
    if args.device in ("cuda", "both"):
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU benchmark.")
        else:
            devices.append("cuda")
    if args.device in ("cpu", "both"):
        devices.append("cpu")

    # 可选：设置 CPU 线程数
    if "cpu" in devices:
        if args.cpu_threads is not None:
            torch.set_num_threads(args.cpu_threads)
            print(f"[cpu] torch threads = {torch.get_num_threads()}")
        else:
            # 启用全部线程
            torch.set_num_threads(torch.get_num_threads())
            print(f"[cpu] torch threads (use all) = {torch.get_num_threads()}")

    # 分别构建 predictor 并测试
    for dev in devices:
        cfg = setup_cfg(args.config_file, args.weights, args.confidence_threshold, dev)
        predictor = DefaultPredictor(cfg)

        # Sanity check
        sanity_check(predictor, image, dev, amp_enabled=(args.amp != "off" and dev == "cuda"))

        # 计时
        results = []
        if dev == "cuda":
            modes = ["off", "on"] if args.amp == "both" else [args.amp]
        else:
            modes = ["off"]  # CPU 默认只测 FP32
        for mode in modes:
            avg = run_predictor(
                predictor,
                image,
                iters=args.iters,
                warmup=args.warmup,
                device=dev,
                amp_enabled=(mode == "on" and dev == "cuda"),
            )
            results.append((mode, avg))

        # 打印结果
        header = "GPU (CUDA)" if dev == "cuda" else "CPU"
        print(f"\n==== Benchmark [{header}] ====")
        for mode, avg in results:
            tag = f"AMP {mode.upper():>3}" if dev == "cuda" else "FP32"
            print(f"{tag}: {avg*1000:.2f} ms / image (iters={args.iters}, warmup={args.warmup})")
        if dev == "cuda" and len(results) == 2:
            off = results[0][1] if results[0][0] == "off" else results[1][1]
            on = results[0][1] if results[0][0] == "on" else results[1][1]
            speedup = off / on if on > 0 else float("inf")
            print(f"Speedup (AMP ON vs OFF): {speedup:.2f}x")


if __name__ == "__main__":
    main()
