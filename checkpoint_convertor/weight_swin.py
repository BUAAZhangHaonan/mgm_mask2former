#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将 Mask2Former (Swin-T) 大权重拆分为三个可独立加载的权重：
- swin_backbone.pth                -> 直接给 `swin.py` 的 SwinTransformer 加载（无 'backbone.' 前缀）
- pixel_decoder_msda.pth           -> 直接给 `msdeformattn.py` 的 MSDeformAttnPixelDecoder 加载（无前缀）
- transformer_decoder_mask2f.pth   -> 直接给 `mask2former_transformer_decoder.py` 的解码器加载（无前缀）

并额外导出一份供 MaskFormerHead 直接加载的合集（注意键名前缀）：
- mask_former_head_submodules.pth  -> 键名以 'pixel_decoder.' 与 'predictor.' 开头（无 'sem_seg_head.'）

默认设置：
- 不重命名 static_query（decoder 在 _load_from_state_dict 内会自动映射）；
- 不删除 Swin 的 relative_position_index（与当前实现/窗口大小吻合）。
"""

import os
import argparse
import torch
import sys
sys.path.insert(0, "/home/fuyx/zhn/mask2former/MGM_Mask2Former")


def _subset_by_prefix(sd, prefix):
    plen = len(prefix)
    return {k[plen:]: v for k, v in sd.items() if k.startswith(prefix)}


def _maybe_drop_rel_pos_index(sd, drop=False):
    if not drop:
        return sd
    keys = [k for k in list(sd.keys()) if k.endswith("relative_position_index")]
    for k in keys:
        sd.pop(k)
    return sd


def _maybe_rename_static_query(sd, do_rename=False):
    if not do_rename:
        return sd
    ren = {}
    for k, v in sd.items():
        if "static_query" in k:
            ren[k.replace("static_query", "query_feat")] = v
        else:
            ren[k] = v
    return ren


def _save(path, sd, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pkg = {"state_dict": sd}
    if meta is not None:
        pkg["meta"] = meta
    torch.save(pkg, path)
    print(f"✓ saved: {path}  ({len(sd)} tensors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to mask2former_coco_swin_t.pth")
    ap.add_argument("--out-dir", type=str, default="split_out",
                    help="directory to write split weights")
    # flags set per your current project:
    ap.add_argument("--rename-static-query", action="store_true",
                    help="rename predictor.static_query -> predictor.query_feat during split (default OFF; decoder会自动映射)")
    ap.add_argument("--drop-rel-pos-index", action="store_true",
                    help="drop Swin relative_position_index buffers (default OFF)")
    # optional: also dump versions with original prefixes kept
    ap.add_argument("--also-keep-prefix", action="store_true",
                    help="besides pure module weights, also dump *_with_prefix.pth keeping original prefixes")
    # quick smoke test (optional)
    ap.add_argument("--test-load", action="store_true",
                    help="after export, try to load each split into its module class (requires your modules importable)")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # 允许两种顶层结构：直接 state_dict / or 包了一层
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt  # 已是 state_dict
    print(f"✓ loaded. total tensors in state_dict: {len(sd)}")

    # ---------- 1) SWIN BACKBONE ----------
    bb_full = _subset_by_prefix(sd, "backbone.")
    bb_noprefix = dict(bb_full)  # copy
    # 是否删除相对位置索引
    bb_noprefix = _maybe_drop_rel_pos_index(bb_noprefix, args.drop_rel_pos_index)
    swin_out = os.path.join(args.out_dir, "swin_backbone.pth")
    _save(
        swin_out,
        bb_noprefix,
        meta={"source": args.ckpt, "module": "SwinTransformer (no prefix)"}
    )
    if args.also_keep_prefix:
        _save(
            os.path.join(args.out_dir, "swin_backbone_with_prefix.pth"),
            _maybe_drop_rel_pos_index(dict(bb_full), args.drop_rel_pos_index),
            meta={"source": args.ckpt, "module": "backbone.* (kept prefix)"}
        )

    # ---------- 2) PIXEL DECODER (MSDeformAttn) ----------
    px_full = _subset_by_prefix(sd, "sem_seg_head.pixel_decoder.")
    px_out = os.path.join(args.out_dir, "pixel_decoder_msda.pth")
    _save(
        px_out,
        px_full,
        meta={"source": args.ckpt, "module": "MSDeformAttnPixelDecoder (no prefix)"}
    )
    if args.also_keep_prefix:
        _save(
            os.path.join(args.out_dir, "pixel_decoder_msda_with_prefix.pth"),
            _subset_by_prefix(sd, "sem_seg_head.pixel_decoder."),
            meta={"source": args.ckpt, "module": "sem_seg_head.pixel_decoder.* (kept prefix)"}
        )

    # ---------- 3) TRANSFORMER DECODER (Mask2Former) ----------
    dec_full = _subset_by_prefix(sd, "sem_seg_head.predictor.")
    # 是否预先把 static_query -> query_feat（默认不改名，交给模块自动处理）
    dec_full = _maybe_rename_static_query(dec_full, args.rename_static_query)
    dec_out = os.path.join(args.out_dir, "transformer_decoder_mask2f.pth")
    _save(
        dec_out,
        dec_full,
        meta={"source": args.ckpt, "module": "MultiScaleMaskedTransformerDecoder (no prefix)"}
    )
    if args.also_keep_prefix:
        keep = _subset_by_prefix(sd, "sem_seg_head.predictor.")
        keep = _maybe_rename_static_query(keep, args.rename_static_query)
        _save(
            os.path.join(args.out_dir, "transformer_decoder_mask2f_with_prefix.pth"),
            keep,
            meta={"source": args.ckpt, "module": "sem_seg_head.predictor.* (kept prefix)"}
        )

    # ---------- 2.5) HEAD BUNDLE: for MaskFormerHead ----------
    # 组合 pixel_decoder + predictor，键名前缀分别为 'pixel_decoder.' 与 'predictor.'（无 'sem_seg_head.'）
    sd_head = {}
    for k, v in px_full.items():
        sd_head[f"pixel_decoder.{k}"] = v
    for k, v in dec_full.items():
        if k.startswith("predictor."):
            sd_head[k] = v
        else:
            sd_head[f"predictor.{k}"] = v

    head_out = os.path.join(args.out_dir, "mask_former_head_submodules.pth")
    _save(
        head_out,
        sd_head,
        meta={
            "source": args.ckpt,
            "module": "MaskFormerHead (pixel_decoder.* + predictor.*)",
            "note": "keys are relative to head; no 'sem_seg_head.' prefix"
        }
    )
    if args.also_keep_prefix:
        # 需要保留 'sem_seg_head.' 的整头版本（用于整模型直接加载）
        sd_head_keep = {}
        for k, v in px_full.items():
            sd_head_keep[f"sem_seg_head.pixel_decoder.{k}"] = v
        raw_pred = _subset_by_prefix(sd, "sem_seg_head.predictor.")
        raw_pred = _maybe_rename_static_query(raw_pred, args.rename_static_query)
        for k, v in raw_pred.items():
            sd_head_keep[f"sem_seg_head.predictor.{k}"] = v
        _save(
            os.path.join(args.out_dir, "mask_former_head_submodules_with_sem_seg_head_prefix.pth"),
            sd_head_keep,
            meta={"source": args.ckpt, "module": "sem_seg_head.(pixel_decoder|predictor).* (kept prefix)"}
        )

    # ---------- Optional: smoke test ----------
    if args.test_load:
        print("\n=== Smoke test: try loading each split ===")
        failed = False
        # 1) Swin
        try:
            from mask2former.modeling.backbone.swin import SwinTransformer
            m = SwinTransformer(in_chans=3)
            missing, unexpected = m.load_state_dict(
                torch.load(swin_out, map_location="cpu", weights_only=True)["state_dict"],
                strict=False
            )
            print(f"[Swin] loaded. missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[Swin] load failed: {e}")
            failed = True
        # 2) Pixel decoder（只做关键键检查）
        try:
            px_sd = torch.load(px_out, map_location="cpu", weights_only=True)["state_dict"]
            must_px = [
                "input_proj.0.0.weight",             # 1x1 conv
                "transformer.level_embed",           # 3x256
                "transformer.encoder.layers.0.self_attn.value_proj.weight",
                "mask_features.weight",              # 1x1 conv
            ]
            miss_px = [k for k in must_px if k not in px_sd]
            if miss_px:
                raise RuntimeError(f"pixel decoder missing keys: {miss_px[:3]} ...")
            print(f"[PixelDecoder] ok: {len(px_sd)} tensors; key sanity checks pass.")
        except Exception as e:
            print(f"[PixelDecoder] check failed: {e}")
            failed = True
        # 3) Transformer decoder
        try:
            from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
                MultiScaleMaskedTransformerDecoder
            )
            dec_sd = torch.load(dec_out, map_location="cpu", weights_only=True)["state_dict"]
            dummy = MultiScaleMaskedTransformerDecoder(
                in_channels=256, num_classes=80, hidden_dim=256, num_queries=100,
                nheads=8, dim_feedforward=2048, dec_layers=9, pre_norm=False,
                mask_dim=256, enforce_input_project=False, mask_classification=True
            )
            missing, unexpected = dummy.load_state_dict(dec_sd, strict=False)
            print(f"[Decoder] loaded. missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[Decoder] load failed: {e}")
            failed = True

        # 4) Head bundle (pixel_decoder.* + predictor.*)
        try:
            head_sd = torch.load(head_out, map_location="cpu", weights_only=True)["state_dict"]
            assert any(k.startswith("pixel_decoder.") for k in head_sd), "missing pixel_decoder.*"
            assert any(k.startswith("predictor.") for k in head_sd), "missing predictor.*"
            must_keys = [
                "pixel_decoder.input_proj.0.0.weight",
                "predictor.class_embed.weight",
                "predictor.mask_embed.layers.0.weight",
            ]
            missing_keys = [k for k in must_keys if k not in head_sd]
            if missing_keys:
                raise RuntimeError(f"missing keys in head bundle: {missing_keys[:3]} ...")
            print(f"[HeadBundle] ok: {len(head_sd)} tensors; prefixes present.")
        except Exception as e:
            print(f"[HeadBundle] check failed: {e}")
            failed = True

        # 5) with_prefix 版本（如果有导出）
        if args.also_keep_prefix:
            try:
                pref = os.path.join(args.out_dir, "mask_former_head_submodules_with_sem_seg_head_prefix.pth")
                pref_sd = torch.load(pref, map_location="cpu", weights_only=True)["state_dict"]
                assert any(k.startswith("sem_seg_head.pixel_decoder.") for k in pref_sd), "missing sem_seg_head.pixel_decoder.*"
                assert any(k.startswith("sem_seg_head.predictor.") for k in pref_sd), "missing sem_seg_head.predictor.*"
                print(f"[HeadBundle (with sem_seg_head.)] ok: {len(pref_sd)} tensors; prefixes present.")
            except Exception as e:
                print(f"[HeadBundle(with_prefix)] check failed: {e}")
                failed = True

        if not failed:
            print("✓ Smoke tests finished.")
        else:
            print("⚠ Some tests failed. Usually due to dummy cfg mismatch; load inside your project for real check.")


if __name__ == "__main__":
    main()
