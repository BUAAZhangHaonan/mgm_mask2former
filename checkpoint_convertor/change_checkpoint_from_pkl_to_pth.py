import pickle
import torch
import numpy as np
from collections import OrderedDict

# --- 步骤 1: 加载 .pkl 文件 ---
pkl_file_path = '/home/remote1/zhanghaonan/projects/mask2former_0822/test/convnext-checkpoint/mask2former_coco_swin_t.pkl'
print(f"Loading checkpoint: {pkl_file_path}")

# 使用 pickle 加载
with open(pkl_file_path, 'rb') as f:
    # 设定 encoding='latin1' 可以在不同 Python 版本间提供更好的兼容性
    data = pickle.load(f, encoding='latin1')

# 提取模型的状态字典
state_dict_numpy = data['model']
print(f"Successfully loaded .pkl file. Found {len(state_dict_numpy)} parameters under the 'model' key.")

# --- 步骤 2: 将 numpy.ndarray 转换为 torch.Tensor ---
new_state_dict = OrderedDict()
for key, value in state_dict_numpy.items():
    if isinstance(value, np.ndarray):
        new_state_dict[key] = torch.from_numpy(value)
    else:
        # 如果还有其他类型的数据，根据情况处理
        new_state_dict[key] = value

print("\nConverted all numpy.ndarray weights to torch.Tensor.")

# --- 步骤 3: 将转换后的状态字典保存为 .pth 文件 ---
pth_output_path = '/home/remote1/zhanghaonan/projects/mask2former_0822/test/convnext-checkpoint/mask2former_coco_swin_t.pth'
torch.save(new_state_dict, pth_output_path)
print(f"Successfully saved the new state dictionary to: {pth_output_path}")

# --- 步骤 4 (验证): 加载新的 .pth 文件并打印结构 ---
print("\nVerifying the newly created .pth file...")
loaded_pth = torch.load(pth_output_path, map_location='cpu')

print("=== Checkpoint Top-level Keys ===")
for key in loaded_pth.keys():
    if isinstance(loaded_pth[key], torch.Tensor):
        print(f"  {key}: {loaded_pth[key].size()}")
    else:
        # 打印非张量的值类型
        print(f"  {key}: <{type(loaded_pth[key]).__name__}>")

print("\nVerification complete. The .pth file is ready to be used with PyTorch and displays torch.Size.")