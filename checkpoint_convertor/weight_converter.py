import torch
from convnext_depth_backbone import ConvNeXtDepthBackbone


def load_and_convert_weights(checkpoint_path, output_path):
    """
    转换新的官方权重 (convnext_tiny_22k_224.pth)
    """
    print("Loading official ConvNeXt checkpoint...")
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)

    original_state_dict = checkpoint['model']

    print("Converting weights...")
    new_state_dict = {}

    for key, value in original_state_dict.items():
        # 特殊处理第一层卷积 (3通道→1通道)
        if key == 'downsample_layers.0.0.weight' and value.shape[1] == 3:
            # 取RGB三通道的平均值
            new_value = value.mean(dim=1, keepdim=True)
            new_state_dict[key] = new_value
            print(f"✓ Converted {key}: {value.shape} → {new_value.shape}")

        # 跳过分类头和全局norm
        elif key.startswith('head.') or key == 'norm.weight' or key == 'norm.bias':
            print(f"✗ Skipping: {key}")
            continue

        # 保留所有其他权重
        else:
            new_state_dict[key] = value

    print(f"\nTotal converted parameters: {len(new_state_dict)}")

    # 保存转换后的权重
    torch.save({
        'state_dict': new_state_dict,
        'meta': {
            'converted_from': checkpoint_path,
            'input_channels': 1,
            'architecture': 'ConvNeXt-Tiny-Depth-22K',
            'original_training': 'ImageNet-22K'
        }
    }, output_path)

    print(f"Converted weights saved to: {output_path}")
    return new_state_dict


def test_weight_loading(converted_path:str):
    """测试权重加载和前向传播"""
    print("\n" + "="*50)
    print("Testing weight loading and forward pass...")

    # 1. 创建模型
    model = ConvNeXtDepthBackbone(in_chans=1)
    print(f"✓ Model created")

    # 2. 加载转换后的权重
    converted_weights = torch.load(
        converted_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(
        converted_weights['state_dict'], strict=False)

    print(f"✓ Weights loaded")
    print(
        f"  Missing keys: {len(missing)} - {missing[:3]}..." if missing else "  No missing keys")
    print(
        f"  Unexpected keys: {len(unexpected)} - {unexpected[:3]}..." if unexpected else "  No unexpected keys")

    # 3. 测试前向传播
    model.eval()
    with torch.no_grad():
        # 模拟深度图输入 (batch_size=2, channels=1, height=224, width=224)
        test_input = torch.randn(2, 1, 224, 224)
        print(f"✓ Test input shape: {test_input.shape}")

        # 前向传播
        outputs = model(test_input)

        print(f"✓ Forward pass successful!")
        print("Output features:")
        for name, tensor in outputs.items():
            print(f"  {name}: {tensor.shape}")

        # 验证输出shape是否符合期望
        expected_shapes = {
            "res2": (2, 96, 56, 56),   # H/4, W/4
            "res3": (2, 192, 28, 28),  # H/8, W/8
            "res4": (2, 384, 14, 14),  # H/16, W/16
            "res5": (2, 768, 7, 7)     # H/32, W/32
        }

        all_correct = True
        for name, expected_shape in expected_shapes.items():
            actual_shape = outputs[name].shape
            if actual_shape == expected_shape:
                print(f"  ✓ {name}: {actual_shape} (correct)")
            else:
                print(f"  ✗ {name}: {actual_shape} (expected {expected_shape})")
                all_correct = False

        if all_correct:
            print("🎉 All output shapes are correct!")
        else:
            print("❌ Some output shapes are incorrect!")

    return model, outputs


if __name__ == "__main__":
    # 转换权重
    original_path = "test/convnext-checkpoint/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth"
    converted_path = "test/convnext-checkpoint/convnext_tiny_depth_22k_converted.pth"

    # 执行转换
    load_and_convert_weights(original_path, converted_path)

    # 测试加载和前向传播
    model, outputs = test_weight_loading(converted_path)
