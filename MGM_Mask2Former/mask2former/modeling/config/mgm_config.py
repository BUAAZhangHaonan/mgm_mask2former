# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_mgm_config(cfg: CN):
    """
    单独用于 RGB-D 实验的 config
    已合并 Mask2Former + Swin + PixelDecoder + MGM + 数据增强/深度处理 所需键。
    """
    # -------------------------
    # 基础求解器配置
    # -------------------------
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.FINETUNE_WEIGHTS = ""

    # -------------------------
    # Mask2Former 核心配置
    # -------------------------
    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MGMMultiScaleMaskedTransformerDecoder"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # Mask2Former 测试配置
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Mask2Former 训练配置
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # -------------------------
    # MGMHead 配置
    # -------------------------
    cfg.MODEL.SEM_SEG_HEAD = CN()
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MGMHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MGMMSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4

    # -------------------------
    # Swin Transformer 配置
    # -------------------------
    cfg.MODEL.RGB_BACKBONE = CN()
    cfg.MODEL.RGB_BACKBONE.NAME = "D2SwinTransformer"
    cfg.MODEL.RGB_BACKBONE.WEIGHTS = ""
    cfg.MODEL.RGB_BACKBONE.SWIN = CN()
    cfg.MODEL.RGB_BACKBONE.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.RGB_BACKBONE.SWIN.PATCH_SIZE = 4
    cfg.MODEL.RGB_BACKBONE.SWIN.EMBED_DIM = 96
    cfg.MODEL.RGB_BACKBONE.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.RGB_BACKBONE.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.RGB_BACKBONE.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.RGB_BACKBONE.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.RGB_BACKBONE.SWIN.QKV_BIAS = True
    cfg.MODEL.RGB_BACKBONE.SWIN.QK_SCALE = None
    cfg.MODEL.RGB_BACKBONE.SWIN.DROP_RATE = 0.0
    cfg.MODEL.RGB_BACKBONE.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.RGB_BACKBONE.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.RGB_BACKBONE.SWIN.APE = False
    cfg.MODEL.RGB_BACKBONE.SWIN.PATCH_NORM = True
    cfg.MODEL.RGB_BACKBONE.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RGB_BACKBONE.SWIN.USE_CHECKPOINT = False

    # -------------------------
    # ConvNeXt 配置
    # -------------------------
    cfg.MODEL.DEPTH_BACKBONE = CN()
    cfg.MODEL.DEPTH_BACKBONE.NAME = "ConvNeXtDepthBackbone"
    cfg.MODEL.DEPTH_BACKBONE.WEIGHTS = ""
    cfg.MODEL.DEPTH_BACKBONE.CONVNEXT = CN()
    cfg.MODEL.DEPTH_BACKBONE.CONVNEXT.DEPTHS = [3, 3, 9, 3]
    cfg.MODEL.DEPTH_BACKBONE.CONVNEXT.DIMS = [96, 192, 384, 768]
    cfg.MODEL.DEPTH_BACKBONE.CONVNEXT.DROP_PATH_RATE = 0.0
    cfg.MODEL.DEPTH_BACKBONE.CONVNEXT.LAYER_SCALE = 1e-6

    # -------------------------
    # MGM / DPE / 边界分支配置
    # -------------------------
    cfg.MODEL.DPE = CN()
    cfg.MODEL.DPE.ENABLED = True
    cfg.MODEL.BOUNDARY = CN()
    cfg.MODEL.BOUNDARY.ENABLED = False
    cfg.MODEL.MGM = CN()
    cfg.MODEL.MGM.ENABLED = True
    cfg.MODEL.MGM.SHARED = True
    cfg.MODEL.MGM.RESIDUAL_ALPHA = 0.05
    cfg.MODEL.MGM.LOSS_ENTROPY_W = 0.01
    cfg.MODEL.MGM.TEMP_INIT = 1.5
    cfg.MODEL.MGM.TEMP_FINAL = 1.0
    cfg.MODEL.MGM.TEMP_STEPS = 3000
    cfg.MODEL.MGM.CLAMP_MIN = 0.05
    cfg.MODEL.MGM.CLAMP_MAX = 0.95
    cfg.MODEL.MGM.NOISE_MASK_WEIGHT = 0.0
    cfg.MODEL.MGM.HIDDEN_DIM = 256
    cfg.MODEL.MGM.FEATURE_DIMS = [96, 192, 384, 768]
    cfg.MODEL.MGM.SCALE_KEYS = ["res2","res3","res4","res5"]
    cfg.MODEL.MGM.POST_FUSE_NORM = True

    # MGM 归一化方法配置
    cfg.MODEL.MGM.ROBUST_NORM = CN()
    cfg.MODEL.MGM.ROBUST_NORM.ENABLED = False
    cfg.MODEL.MGM.ROBUST_NORM.METHOD  = "minmax"  # or "quantile"

    # MGM 深度先验参数配置
    cfg.MODEL.MGM.PRIOR = CN()
    cfg.MODEL.MGM.PRIOR.ENABLED = True
    cfg.MODEL.MGM.PRIOR.USE_GRADIENT   = True
    cfg.MODEL.MGM.PRIOR.USE_VARIANCE   = True
    cfg.MODEL.MGM.PRIOR.USE_VALID_HOLE = True
    cfg.MODEL.MGM.PRIOR.USE_RGB_EDGE   = False
    cfg.MODEL.MGM.PRIOR.VAR_KERNEL = 5
    cfg.MODEL.MGM.PRIOR.Z_MIN = 0.0
    cfg.MODEL.MGM.PRIOR.Z_MAX = 1.0
    cfg.MODEL.MGM.PRIOR.COMPUTE_ON = "res3"   # "full"/"res2"/"res3"/"res4"/"res5"

    # RGB Input Pixel Mean/Std
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

    # -------------------------
    # 输入数据配置（RGB-D 专用）
    # -------------------------

    # 数据集根目录
    cfg.INPUT.DATASET_ROOT = None #在训练脚本入口指定

    # LSJ 几何增强（RGB & Depth 同步）
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # ["horizontal", "vertical", "none"]
    cfg.INPUT.SIZE_DIVISIBILITY = 32

    # 模态/mapper
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_instance_lsj_rgbd"

    # RGB 光度增强（仅作用于 RGB）
    cfg.INPUT.RGB_PHOTO_AUG = CN()
    cfg.INPUT.RGB_PHOTO_AUG.ENABLED = False
    cfg.INPUT.RGB_PHOTO_AUG.BRIGHTNESS = 0.0   # [0,1] 建议0.2起
    cfg.INPUT.RGB_PHOTO_AUG.CONTRAST = 0.0   # [0,1]
    cfg.INPUT.RGB_PHOTO_AUG.SATURATION = 0.0   # [0,1]
    cfg.INPUT.RGB_PHOTO_AUG.HUE = 0.0   # [0,0.5]

    # 深度数据读取/归一化/裁剪/噪声
    cfg.INPUT.DEPTH_FORMAT = "I"        # 常见uint16
    cfg.INPUT.DEPTH_SCALE = 0.001      # mm->m 的默认换算
    cfg.INPUT.DEPTH_SHIFT = 0.0        # 有些相机需要整体平移
    cfg.INPUT.DEPTH_CLIP_MIN = 0.0      # 单位：米
    cfg.INPUT.DEPTH_CLIP_MAX = 1.0
    cfg.INPUT.DEPTH_NORM = "minmax"   # ["none", "minmax", [min, max]]

    cfg.INPUT.DEPTH_NOISE = CN()
    cfg.INPUT.DEPTH_NOISE.ENABLED = False
    cfg.INPUT.DEPTH_NOISE.GAUSSIAN_STD = 0.0    # 加性高斯
    cfg.INPUT.DEPTH_NOISE.SPECKLE_STD = 0.0    # 乘性散斑
    cfg.INPUT.DEPTH_NOISE.DROP_PROB = 0.0    # 随机丢深度比例
    cfg.INPUT.DEPTH_NOISE.DROP_VAL = 0.0    # 丢失填充值（0常见）

    # 保留原 mask2former 数据增强开关
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0

    # -------------------------
    # 数据加载器配置
    # -------------------------
    if not hasattr(cfg, 'DATALOADER'):
        cfg.DATALOADER = CN()
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.LOAD_PROPOSALS = False
    cfg.DATALOADER.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
    cfg.DATALOADER.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
    cfg.DATALOADER.PROPOSAL_FILES_TRAIN = ()
    cfg.DATALOADER.PROPOSAL_FILES_TEST = ()
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True

    # -------------------------
    # 测试配置
    # -------------------------
    if not hasattr(cfg.TEST, 'AUG'):
        cfg.TEST.AUG = CN()
        cfg.TEST.AUG.ENABLED = False
        cfg.TEST.AUG.MIN_SIZES = (
            400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
        cfg.TEST.AUG.MAX_SIZE = 4000
        cfg.TEST.AUG.FLIP = True

    cfg.VERSION = 2.0
