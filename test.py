import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from model import UNet

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载模型
def load_model(weight_path):
    model = UNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()  # 设置模型为评估模式
    return model


# 预测特征图
def predict_feature_map(model, image_path, target_size=(1024, 1024)):
    # 加载原图
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # 保存原始尺寸

    # 调整图片大小
    image_resized = image.resize(target_size, Image.BILINEAR)
    image_resized = np.array(image_resized) / 255.0  # 归一化
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

    # 通过模型预测特征图
    with torch.no_grad():
        feature_map = model(image_tensor)  # 模型输出特征图
        feature_map = feature_map.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

    # 将特征图调整回原始尺寸
    feature_map_resized = Image.fromarray(feature_map).resize(original_size, Image.BILINEAR)
    return np.array(feature_map_resized), original_size


# 标注最大值位置
def annotate_max_point(image_path, feature_map):
    # 找到特征图中的最大值位置
    max_pos = np.unravel_index(np.argmax(feature_map), feature_map.shape)
    max_y, max_x = max_pos  # 行列对应坐标 (y, x)

    # 打开原图并标注最大值点
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    point_radius = 10  # 标注点的半径
    point_color = (255, 0, 0)  # 红色
    draw.ellipse(
        [(max_x - point_radius, max_y - point_radius), (max_x + point_radius, max_y + point_radius)],
        fill=point_color,
        outline=point_color
    )
    return image, (max_x, max_y)


# 显示图像
def display_results(image_path, feature_map, annotated_image):
    # 显示原图
    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image (A)")
    plt.imshow(image)
    plt.axis("off")

    # 显示特征图
    plt.subplot(1, 3, 2)
    plt.title("Feature Map (B)")
    plt.imshow(feature_map, cmap='hot')
    plt.colorbar()
    plt.axis("off")

    # 显示标注图
    plt.subplot(1, 3, 3)
    plt.title("Annotated Image (C)")
    plt.imshow(annotated_image)
    plt.axis("off")

    # 展示结果
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 参数设置
    weight_path = r'/data0/user/jjxie/projects/heat_map/weight/weight.pth'  # 模型权重路径
    image_path = r'/data0/user/jjxie/projects/heat_map/ralateddata/jpg/408_left.jpg'  # 输入原图路径

    # 加载模型
    model = load_model(weight_path)

    # 预测特征图
    feature_map, original_size = predict_feature_map(model, image_path, target_size=(1024, 1024))

    # 标注最大值点
    annotated_image, max_point = annotate_max_point(image_path, feature_map)

    # 打印最大值点信息
    print(f"Max Point: {max_point}, Feature Map Max Value: {feature_map.max()}")

    # 显示结果
    display_results(image_path, feature_map, annotated_image)

    # 保存三张图的输出结果
    annotated_image.save('annotated_image.jpg')
    plt.imsave('feature_map.jpg', feature_map, cmap='hot')
    Image.open(image_path).save('original_image.jpg')
    # 三张图被保存在当前目录下，分别为原图、特征图和标注图。