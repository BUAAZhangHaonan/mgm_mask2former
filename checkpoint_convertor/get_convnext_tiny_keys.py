import torch

# 方法1：直接查看
print("==================== 1 ====================")
ckpt = torch.load(
    'test/convnext-checkpoint/convnext_tiny_22k_224.pth', map_location='cpu', weights_only=False)
print("Keys in checkpoint:")
for key in ckpt.keys():
    print(
        f"  {key}: {ckpt[key].shape if torch.is_tensor(ckpt[key]) else type(ckpt[key])}")

# 方法2：如果有嵌套结构
print("==================== 2 ====================")
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

print("\nState dict keys:")
for key in state_dict.keys():
    print(f"  {key}: {state_dict[key].shape}")
