# -*- coding: utf-8 -*-
import os
import random
import shutil



# 设置路径
image_dir = './images'  # 替换为你的图像文件夹路径
label_dir = './labels'  # 替换为你的标签文件夹路径
train_image_dir = './dataset/images/train/'  # 替换为训练图像文件夹路径
train_label_dir = './dataset/labels/train/'  # 替换为训练标签文件夹路径
val_image_dir = './dataset/images/val/'  # 替换为验证图像文件夹路径
val_label_dir = './dataset/labels/val/'  # 替换为验证标签文件夹路径
test_image_dir = './dataset/images/test/'  # 替换为测试图像文件夹路径
test_label_dir = './dataset/labels/test/'  # 替换为测试标签文件夹路径

# 确保目标文件夹存在
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(
    image_dir) if f.endswith('.jpg')]  # 修改为你的图像扩展名
random.shuffle(image_files)

# 按5:2.5:2.5的比例分割数据集
train_split = int(0.5 * len(image_files))
val_split = int(0.75 * len(image_files))
train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# 移动图像和对应的标签文件到训练、验证和测试文件夹


def move_files(files, source_image_dir, source_label_dir, target_image_dir, target_label_dir):
    for file in files:
        image_path = os.path.join(source_image_dir, file)
        label_path = os.path.join(
            source_label_dir, file.replace('.jpg', '.txt'))  # 修改为你的标签扩展名
        shutil.copy(image_path, target_image_dir)
        if os.path.exists(label_path):
            shutil.copy(label_path, target_label_dir)


move_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)
move_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)
move_files(test_files, image_dir, label_dir, test_image_dir, test_label_dir)

print(f"训练集包含 {len(train_files)} 张图像")
print(f"验证集包含 {len(val_files)} 张图像")
print(f"测试集包含 {len(test_files)} 张图像")
