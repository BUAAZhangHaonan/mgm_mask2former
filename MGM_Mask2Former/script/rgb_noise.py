#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corruption shards with mirrored folder layout:

Input dataset (read-only):
  <DATASET_ROOT>/
    annotations/{instances_*.json}
    images/{train,val,test}/...      <-- we mirror THIS subtree

Output shards:
  <DATASET_ROOT>/images_corrupted/<corruption>/s{1..5}/images/{train,val,test}/<same_relpath>.png
"""

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------- utils ----------
def to_float01(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0

def from_float01(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)

def rng_for(relpath: str, name: str, severity: int, base_seed: int) -> np.random.RandomState:
    key = f"{relpath}|{name}|s{severity}|seed{base_seed}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    seed = int(h[:8], 16) ^ base_seed
    return np.random.RandomState(seed)

def odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1

def scale_by_res(v_1024: float, h: int, w: int) -> int:
    s = max(1.0, min(h, w) / 1024.0)
    return max(1, int(round(v_1024 * s)))

def atomic_write_png(path: Path, rgb_u8: np.ndarray) -> bool:
    """
    原子写 PNG：
    - 先在同目录下写入一个临时 .png（隐藏名/.tmp 前缀，但**扩展名仍是 .png**）
    - 成功后用 os.replace 原子替换为目标文件
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 临时文件名：.<stem>.tmp.png  （确保以 .png 结尾，OpenCV 才能写）
    tmp = path.parent / f".{path.stem}.tmp{path.suffix}"   # suffix = ".png"

    try:
        # 方式A：用 imencode 保证 png 编码，然后写字节（兼容性更好）
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        if not ok:
            return False
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())

        # 原子替换
        os.replace(tmp, path)
        return True
    except Exception:
        # 出错清理
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False

# ---------- corruptions ----------
def corrupt_gaussian(x, s, rng):
    sigmas = [0.01, 0.02, 0.04, 0.06, 0.08]
    y = x + rng.normal(0.0, sigmas[s-1], size=x.shape).astype(np.float32)
    return np.clip(y, 0.0, 1.0)

def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    cv2.line(k, (0, length//2), (length-1, length//2), 1.0, 1, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((length/2-0.5, length/2-0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length), flags=cv2.INTER_LINEAR)
    k /= max(k.sum(), 1e-8)
    return k

def corrupt_motion_blur(x, s, rng):
    lengths = [3, 5, 9, 15, 21]
    h, w = x.shape[:2]
    L = odd(scale_by_res(lengths[s-1], h, w))
    k = _motion_kernel(L, rng.uniform(0, 180.0))
    y = np.stack([cv2.filter2D(x[..., c], -1, k, borderType=cv2.BORDER_REFLECT) for c in range(3)], 2)
    return np.clip(y, 0.0, 1.0)

def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    k = np.zeros((2*r+1, 2*r+1), np.float32)
    cy = cx = r
    for y in range(2*r+1):
        for x in range(2*r+1):
            if (y-cy)**2 + (x-cx)**2 <= r*r + 0.25:
                k[y, x] = 1.0
    k /= k.sum()
    return k

def corrupt_defocus_blur(x, s, rng):
    radii = [1, 2, 3, 5, 7]
    h, w = x.shape[:2]
    r = scale_by_res(radii[s-1], h, w)
    k = _disk_kernel(r)
    y = np.stack([cv2.filter2D(x[..., c], -1, k, borderType=cv2.BORDER_REFLECT) for c in range(3)], 2)
    return np.clip(y, 0.0, 1.0)

def corrupt_jpeg(x, s, rng):
    qualities = [95, 85, 70, 50, 30]
    q = int(qualities[s-1])
    u8 = from_float01(x)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return x
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return to_float01(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))

def corrupt_downup(x, s, rng):
    factors = [0.9, 0.75, 0.6, 0.5, 0.4]
    f = factors[s-1]
    h, w = x.shape[:2]
    nh, nw = max(1, int(round(h*f))), max(1, int(round(w*f)))
    small = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(back, 0.0, 1.0)

def corrupt_gamma(x, s, rng):
    gammas = [0.9, 0.8, 0.7, 0.6, 0.5]
    return np.power(np.clip(x, 0.0, 1.0), gammas[s-1], dtype=np.float32)

def corrupt_bright_contrast(x, s, rng):
    b_levels = [0.05, 0.10, 0.15, 0.20, 0.25]
    c_levels = [1.05, 1.10, 1.20, 1.30, 1.40]
    b = b_levels[s-1] * rng.choice([-1.0, 1.0])
    c = c_levels[s-1]
    return np.clip((x + b) * c, 0.0, 1.0)

def corrupt_white_balance(x, s, rng):
    ranges = {
        1: ((0.95, 1.05), (0.98, 1.02), (0.95, 1.05)),
        2: ((0.93, 1.07), (0.97, 1.03), (0.93, 1.07)),
        3: ((0.90, 1.10), (0.96, 1.04), (0.90, 1.10)),
        4: ((0.88, 1.12), (0.95, 1.05), (0.88, 1.12)),
        5: ((0.85, 1.15), (0.95, 1.05), (0.85, 1.15)),
    }
    (rmin, rmax), (gmin, gmax), (bmin, bmax) = ranges[s]
    m = np.array([rng.uniform(rmin, rmax), rng.uniform(gmin, gmax), rng.uniform(bmin, bmax)],
                 dtype=np.float32).reshape(1,1,3)
    return np.clip(x * m, 0.0, 1.0)

def _apply(x, fn_list, s, rng):
    y = x
    for fn in fn_list:
        y = fn(y, s, rng)
    return np.clip(y, 0.0, 1.0)

def corrupt_mixed_all(x, s, rng):
    # 1) blur/resolution: pick exactly 1
    blur_pool = [corrupt_motion_blur, corrupt_defocus_blur, corrupt_downup]
    blur_fn = rng.choice(blur_pool)

    # 2) jpeg probabilistic
    p_jpeg = [0.3, 0.5, 0.7, 0.9, 1.0][s-1]
    use_jpeg = rng.rand() < p_jpeg

    # 3) one photometric
    photo_pool = [corrupt_gamma, corrupt_bright_contrast, corrupt_white_balance]
    photo_fn = rng.choice(photo_pool)

    # 4) gaussian noise probabilistic
    p_noise = [0.3, 0.4, 0.6, 0.8, 1.0][s-1]
    use_noise = rng.rand() < p_noise

    ops = [blur_fn]
    if use_jpeg: ops.append(corrupt_jpeg)
    if use_noise: ops.append(corrupt_gaussian)
    ops.append(photo_fn)

    return _apply(x, ops, s, rng)

def corrupt_mixed_blur(x, s, rng):
    blur_pool = [corrupt_motion_blur, corrupt_defocus_blur, corrupt_downup]
    blur_fn = rng.choice(blur_pool)
    p_jpeg = [0.3, 0.5, 0.7, 0.9, 1.0][s-1]
    ops = [blur_fn] + ([corrupt_jpeg] if (rng.rand() < p_jpeg) else [])
    return _apply(x, ops, s, rng)

def corrupt_mixed_photometric(x, s, rng):
    photo_pool = [corrupt_gamma, corrupt_bright_contrast, corrupt_white_balance]
    # pick 2 distinct photometric ops
    idx = rng.choice(len(photo_pool), size=2, replace=False)
    ops = [photo_pool[i] for i in idx]

    p_noise = [0.3, 0.4, 0.6, 0.8, 1.0][s-1]
    if rng.rand() < p_noise:
        ops.insert(0, corrupt_gaussian)

    p_jpeg = [0.3, 0.5, 0.7, 0.9, 1.0][s-1]
    if rng.rand() < p_jpeg:
        ops.insert(0, corrupt_jpeg)
    return _apply(x, ops, s, rng)

CORRUPTIONS = {
    "gaussian": corrupt_gaussian,
    "motion_blur": corrupt_motion_blur,
    "defocus_blur": corrupt_defocus_blur,
    "jpeg": corrupt_jpeg,
    "downup": corrupt_downup,
    "gamma": corrupt_gamma,
    "bright_contrast": corrupt_bright_contrast,
    "white_balance": corrupt_white_balance,
    "mixed_all": corrupt_mixed_all,
    "mixed_blur": corrupt_mixed_blur,
    "mixed_photometric": corrupt_mixed_photometric
}

# ---------- pipeline ----------
def list_split_images(images_root: Path, split: str) -> List[Path]:
    root = images_root / split
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def process_one(img_path: Path, rel_from_split: Path, out_base: Path,
                name: str, severity: int, seed: int, overwrite: bool) -> str:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return f"[skip] cannot read: {img_path}"
    x = to_float01(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Mirror layout: <out_base>/images/<split>/<rel>.png
    out_path = (out_base / "images" / rel_from_split).with_suffix(".png")
    if (not overwrite) and out_path.exists():
        return f"[ok] exists: {out_path}"

    rng = rng_for(str(rel_from_split).replace(os.sep, "/"), name, severity, seed)
    y = CORRUPTIONS[name](x, severity, rng)
    y_u8 = from_float01(y)
    ok = atomic_write_png(out_path, y_u8)
    return f"[ok] write: {out_path}" if ok else f"[fail] write: {out_path}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--splits", type=str, nargs="+", default=["val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--corruptions", type=str, nargs="+", default=["all"],
                    choices=list(CORRUPTIONS.keys()) + ["all"])
    ap.add_argument("--severities", type=int, nargs="+", default=[1,2,3,4,5],
                    choices=[1,2,3,4,5])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--out-subdir", type=str, default="images_corrupted")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds_root = Path(args.dataset_root).resolve()
    images_root = ds_root / "images"
    assert images_root.exists(), f"not found: {images_root}"

    names = list(CORRUPTIONS.keys()) if ("all" in args.corruptions) else args.corruptions
    severities = sorted(set(args.severities))

    jobs = []
    for split in args.splits:
        split_imgs = list_split_images(images_root, split)
        print(f"[info] split={split} images={len(split_imgs)}")
        for img in split_imgs:
            rel_from_split = Path(split) / img.relative_to(images_root / split)
            for name in names:
                for s in severities:
                    # shard base = <root>/images_corrupted/<corruption>/sX
                    out_base = ds_root / args.out_subdir / name / f"s{s}"
                    jobs.append((img, rel_from_split, out_base, name, s))

    print(f"[info] total jobs: {len(jobs)}  ->  {ds_root / args.out_subdir}")

    def worker(job):
        img, rel, out_base, name, s = job
        return process_one(img, rel, out_base, name, s, args.seed, args.overwrite)

    with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for f in as_completed(futures):
            _ = f.result()
    print("[done]")

if __name__ == "__main__":
    main()
