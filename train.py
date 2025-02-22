import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from dataset import MyDataset
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import time

# 手动选择设备（优先使用 GPU 1，如果不可用则使用其他 GPU 或 CPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 默认选择 GPU 1
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    device = torch.device('cpu')
    print("CUDA not available. Using CPU.")
else:
    device_id = 0  # 选择当前可用 GPU 的编号
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        if torch.cuda.get_device_properties(i).total_memory > 4 * 1024**3:  # 如果显存大于 4GB，选择此 GPU
            device_id = i
            break
    device = torch.device(f'cuda:{device_id}')
    print(f"Using GPU: {torch.cuda.get_device_name(device_id)} (Device ID: {device_id})")

# 权重路径和数据路径
weight_path = r'/data0/user/jjxie/projects/heat_map/weight/weight1024_3.pth'
data_path = r'/data0/user/jjxie/projects/heat_map/ralateddata'

if __name__ == '__main__':
    # 初始化数据集和 DataLoader
    dataset = MyDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=6, shuffle=True)

    # 初始化网络
    net = UNet().to(device)
    # # 睡眠一分钟
    # print("睡眠一分钟,请观察显存")
    # time.sleep(20)
    # print("睡眠结束，开始训练")
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print("成功加载权重文件")
    else:
        print("未能找到权重文件，使用随机初始化的模型")

    # 优化器和损失函数
    opt = optim.Adam(net.parameters(), lr=1e-4)
    loss_function = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss
    scaler = GradScaler()  # 混合精度训练

    max_epoch = 150

    # 开始训练
    for epoch in range(1, max_epoch + 1):
        print(f"Epoch {epoch} Start")
        epoch_loss = 0.0
        torch.cuda.empty_cache()  # 清理显存缓存

        with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", unit="batch") as pbar:
            for i, (image, heatmap) in enumerate(data_loader):
                image = image.to(device)
                heatmap = heatmap.to(device)

                # 使用混合精度前向传播
                with autocast():
                    out_image = net(image)  # 模型输出 (无 Sigmoid)
                    train_loss = loss_function(out_image, heatmap)  # 使用 BCEWithLogitsLoss

                # 混合精度反向传播
                scaler.scale(train_loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                epoch_loss += train_loss.item()
                pbar.set_postfix({"loss": train_loss.item()})
                pbar.update(1)

        # 保存权重
        if epoch % 3 == 0:
            torch.save(net.state_dict(), weight_path)
            print(f"权重已保存到 {weight_path}")

        print(f"Epoch {epoch} Loss: {epoch_loss / len(data_loader):.4f}")

    print("训练完成!")
