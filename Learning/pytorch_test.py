import torch

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    print(f"XPU是否可用: {torch.xpu.is_available()}")
    print(f"可用XPU设备数: {torch.xpu.device_count()}")
    if torch.xpu.is_available():
        print(f"设备名称: {torch.xpu.get_device_name(0)}")

    # 测试简单计算
    if torch.xpu.is_available():
        x = torch.randn(2, 3).xpu()
        y = torch.randn(3, 2).xpu()
        z = torch.matmul(x, y)
        print(f"矩阵乘法结果: {z}")
        print(f"计算设备: {z.device}")
    else:
        print("XPU不可用，请检查安装")
