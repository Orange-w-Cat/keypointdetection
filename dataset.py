from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from heatmap_label import CenterLabelHeatMap


class MyDataset(Dataset):
    def __init__(self, root):
        self.path = root  # 数据集路径
        self.img_name_list = os.listdir(os.path.join(root, 'jpg'))  # 获取所有图片名
        txt_file_path = os.path.join(self.path, 'txt', 'ele_data.txt')
        self.file_names, self.coordinates = self.read_txt(txt_file_path)

        # 过滤图片，仅保留在 TXT 文件中存在的图片
        self.img_name_list = [
            img_name for img_name in self.img_name_list
            if os.path.splitext(img_name)[0] in self.file_names
        ]

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        # 获取图片路径和文件名
        image_name = self.img_name_list[index]
        image_path = os.path.join(self.path, 'jpg', image_name)
        image_name_no_ext = os.path.splitext(image_name)[0]

        # 找到对应的坐标
        coordinate_index = self.file_names.index(image_name_no_ext)
        coordinates = self.coordinates[coordinate_index]

        # 加载图片并确保其为 3 通道 RGB 格式
        image = Image.open(image_path).convert("RGB")  # 强制转换为 RGB

        # 调整图片大小到 1024x1024
        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)

        image = np.array(image) / 255.0  # 归一化到 [0, 1]
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 转换为 [C, H, W]

        # 生成热力图
        heat_image = CenterLabelHeatMap(
            img_width=image_tensor.shape[2],
            img_height=image_tensor.shape[1],
            c_x=coordinates[0],
            c_y=coordinates[1],
            sigma=13
        )
        heat_image = torch.tensor(heat_image, dtype=torch.float32).unsqueeze(0)  # 单通道

        return image_tensor, heat_image

    def read_txt(self, txt_path):
        """
        从 TXT 文件中读取文件名和坐标。
        假设每行格式为："101_left.json: [604.77, 813.34]"
        """
        file_names = []
        coordinates = []

        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(': ')
                if len(parts) != 2:
                    continue
                file_name = parts[0].replace('.json', '').strip()
                coordinate = parts[1].strip('[]')
                x, y = map(float, coordinate.split(','))
                file_names.append(file_name)
                coordinates.append([x, y])

        return file_names, coordinates


if __name__ == '__main__':
    dataset = MyDataset(r'/data0/user/jjxie/projects/heat_map/ralateddata')
    print(f"Dataset size: {len(dataset)}")

    # 验证前 10 个样本的输出
    for i in range(min(10, len(dataset))):
        try:
            image, heatmap = dataset[i]
            print(f"Sample {i}:")
            print(f"  Image tensor shape: {image.shape}")  # 应为 [3, 1024, 1024]
            print(f"  Heatmap tensor shape: {heatmap.shape}")  # 应为 [1, 1024, 1024]）
            print(f"  Image max value: {image.max()}, min value: {image.min()}")  # 检查归一化范围
            print(f"  Heatmap max value: {heatmap.max()}, min value: {heatmap.min()}")  # 检查热力图值
        except Exception as e:
            print(f"Error in sample {i}: {e}")
