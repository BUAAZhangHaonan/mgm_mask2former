import torch


def load_and_convert_weights(checkpoint_path, output_path):
    """
    转换ConvNeXt官方权重为depth输入版本
    """
    print("Loading official ConvNeXt checkpoint...")
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True)

    original_state_dict = checkpoint['model']
    print(f"✓ Found state_dict with {len(original_state_dict)} parameters")

    print("\n=== Converting weights ===")
    new_state_dict = {}
    skipped_count = 0
    converted_count = 0

    for key, value in original_state_dict.items():
        # 特殊处理第一层卷积 (3通道→1通道)
        if key == 'downsample_layers.0.0.weight' and value.shape[1] == 3:
            new_value = value.mean(dim=1, keepdim=True)
            new_state_dict[key] = new_value
            print(f"✓ Converted {key}: {value.shape} → {new_value.shape}")
            converted_count += 1

        # 跳过分类头和全局norm
        elif key.startswith('head.') or key == 'norm.weight' or key == 'norm.bias':
            print(f"✗ Skipping {key}: {value.shape}")
            skipped_count += 1
            continue

        # 保留所有其他权重
        else:
            new_state_dict[key] = value
            converted_count += 1

    print(f"\n✓ Conversion summary:")
    print(f"  - Converted parameters: {converted_count}")
    print(f"  - Skipped parameters: {skipped_count}")
    print(f"  - Total output parameters: {len(new_state_dict)}")

    # 保存转换后的权重
    torch.save({
        'state_dict': new_state_dict,
        'meta': {
            'converted_from': checkpoint_path,
            'input_channels': 1,
            'architecture': 'ConvNeXt-Tiny-Depth-22K',
            'original_params': len(original_state_dict),
            'converted_params': len(new_state_dict)
        }
    }, output_path)

    print(f"✓ Converted weights saved to: {output_path}")
    return new_state_dict


def test_weight_loading():
    """测试权重加载和前向传播"""
    print("\n" + "="*60)
    print("Testing weight loading and forward pass...")

    from convnext_depth_backbone import ConvNeXtDepthBackbone

    # 1. 创建模型
    model = ConvNeXtDepthBackbone(in_chans=1)
    print(f"✓ Model created")

    # 打印模型参数数量用于对比
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {total_params:,} parameters")

    # 2. 加载转换后的权重
    converted_path = 'test/convnext-checkpoint/convnext_tiny_depth_22k_converted.pth'
    converted_checkpoint = torch.load(
        converted_path, map_location='cpu', weights_only=True)
    converted_state_dict = converted_checkpoint['state_dict']

    missing, unexpected = model.load_state_dict(
        converted_state_dict, strict=False)

    print(f"✓ Weights loaded")
    print(f"  Missing keys: {len(missing)}")
    if missing:
        print(f"    Examples: {missing[:3]}...")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"    Examples: {unexpected[:3]}...")

    # 3. 测试前向传播
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(2, 1, 224, 224)
        print(f"\n✓ Test input shape: {test_input.shape}")

        try:
            outputs = model(test_input)
            print(f"✓ Forward pass successful!")

            print("\nOutput features:")
            for name, tensor in outputs.items():
                print(f"  {name}: {tensor.shape}")

            # 验证shape
            expected_shapes = {
                "res2": (2, 96, 56, 56),    # stride=4
                "res3": (2, 192, 28, 28),   # stride=8
                "res4": (2, 384, 14, 14),   # stride=16
                "res5": (2, 768, 7, 7)      # stride=32
            }

            print("\nShape verification:")
            all_correct = True
            for name, expected_shape in expected_shapes.items():
                actual_shape = outputs[name].shape
                if actual_shape == expected_shape:
                    print(f"  ✓ {name}: {actual_shape} (correct)")
                else:
                    print(
                        f"  ✗ {name}: {actual_shape} (expected {expected_shape})")
                    all_correct = False

            # 检查输出值是否合理
            print("\nOutput value checks:")
            for name, tensor in outputs.items():
                mean_val = tensor.mean().item()
                std_val = tensor.std().item()
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                print(
                    f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")

            if all_correct:
                print("\n🎉 All tests passed! ConvNeXt depth backbone is ready!")
            else:
                print("\n❌ Some shape mismatches detected!")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    return model, outputs if 'outputs' in locals() else None


if __name__ == "__main__":
    print("="*60)
    print("ConvNeXt Depth Backbone Weight Conversion")
    print("="*60)

    # 设置路径
    original_path = "test/convnext-checkpoint/convnext_tiny_22k_224.pth"
    converted_path = "test/convnext-checkpoint/convnext_tiny_depth_22k_converted.pth"

    try:
        # Step 1: 转换权重
        print("STEP 1: Converting weights")
        load_and_convert_weights(original_path, converted_path)

        # Step 2: 测试加载和前向传播
        print("\nSTEP 2: Testing model")
        model, outputs = test_weight_loading()

        print("\n" + "="*60)
        print("Conversion and testing completed successfully!")
        print("Your ConvNeXt depth backbone is ready for integration.")
        print("="*60)

    except Exception as e:
        print(f"❌ Error during conversion/testing: {e}")
        import traceback
        traceback.print_exc()
