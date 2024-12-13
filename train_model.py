from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./yolo11m.pt")
    model.train(data='data.yaml', epochs=80, imgsz=640,
                device='cuda:0', batch=8, workers=6)
    # 基于yolo11m进行续训练 训练轮次 80 单此 训练图片大小640张 使用第一张显卡
