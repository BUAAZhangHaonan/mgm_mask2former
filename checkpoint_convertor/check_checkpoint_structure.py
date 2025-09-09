import torch
import os


def check_checkpoint_structure(checkpoint_path):
    """检查checkpoint的实际结构"""
    print(f"Loading checkpoint: {checkpoint_path}")

    # 首先检查文件是否存在和基本信息
    if not os.path.exists(checkpoint_path):
        print(f"Error: File does not exist: {checkpoint_path}")
        return None, None

    file_size = os.path.getsize(checkpoint_path)
    print(f"File size: {file_size / (1024*1024):.2f} MB")

    # 检查文件扩展名
    file_ext = os.path.splitext(checkpoint_path)[1].lower()
    print(f"File extension: {file_ext}")

    # 首先尝试使用weights_only=True（推荐的安全方式）
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("Successfully loaded with weights_only=True")
    except Exception as e:
        print(f"Failed to load with weights_only=True: {e}")
        print("Falling back to weights_only=False...")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            print("Successfully loaded with weights_only=False")
        except Exception as e2:
            print(f"Failed to load checkpoint with weights_only=False: {e2}")

            # 尝试其他加载方式
            print("Trying alternative loading methods...")

            # 尝试不指定weights_only参数
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                print("Successfully loaded without weights_only parameter")
            except Exception as e3:
                print(f"Failed to load without weights_only: {e3}")

                # 尝试使用pickle直接加载
                try:
                    import pickle

                    with open(checkpoint_path, "rb") as f:
                        checkpoint = pickle.load(f)
                    print("Successfully loaded with pickle")
                except Exception as e4:
                    print(f"Failed to load with pickle: {e4}")

                    # 检查文件开头的magic number
                    try:
                        with open(checkpoint_path, "rb") as f:
                            magic = f.read(8)
                            print(f"File magic bytes: {magic.hex()}")
                    except Exception as e5:
                        print(f"Failed to read file magic: {e5}")

                    return None, None

    print("\n=== Checkpoint Top-level Keys ===")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  {key}: <dict with {len(value)} items>")
        elif torch.is_tensor(value):
            print(f"  {key}: tensor {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # 尝试找到实际的state_dict
    possible_state_dict_keys = ["model", "state_dict", "model_state_dict"]
    state_dict = None
    state_dict_key = None

    for key in possible_state_dict_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            state_dict_key = key
            break

    if state_dict is None:
        # 如果没有找到，可能整个checkpoint就是state_dict
        print("\n=== Trying to use checkpoint as state_dict directly ===")
        state_dict = checkpoint
        state_dict_key = "direct"

    print(f"\n=== Using '{state_dict_key}' as state_dict ===")
    print(f"State dict contains {len(state_dict)} parameters")

    # 显示前几个权重键名
    # print("\n=== First 10 Parameter Keys ===")
    for _, key in enumerate(list(state_dict.keys())):
        value = state_dict[key]
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    return state_dict, state_dict_key


# 测试
# checkpoint_path = "/home/remote1/zhanghaonan/projects/mask2former_0825/test/convnext-checkpoint/convnext_tiny_depth_22k_converted.pth"  # 你的权重文件路径
# checkpoint_path = "/home/remote1/zhanghaonan/projects/mask2former_0822/test/convnext-checkpoint/mask2former_coco_swin_t.pkl"  # 你的权重文件路径
checkpoint_path = (
    "MGM_Mask2Former/pretrained-checkpoint/0909_2K_REAL_0909_5K.pth"  # 你的权重文件路径
)
state_dict, key_used = check_checkpoint_structure(checkpoint_path)

if state_dict is None:
    print("\n=== Failed to load checkpoint! ===")
    print("The checkpoint file might be corrupted or in an unsupported format.")

    # 建议尝试其他文件
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(f"\nTrying to list other checkpoint files in {checkpoint_dir}:")
    try:
        files = os.listdir(checkpoint_dir)
        checkpoint_files = [
            f for f in files if f.endswith(".pkl") or f.endswith(".pth")
        ]
        for f in checkpoint_files:
            full_path = os.path.join(checkpoint_dir, f)
            size = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  {f}: {size:.2f} MB")
    except Exception as e:
        print(f"Failed to list directory: {e}")
else:
    print(f"\nCheckpoint loaded successfully using key: {key_used}")
