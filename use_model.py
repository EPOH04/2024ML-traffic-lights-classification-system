from ultralytics import YOLO
import os
model = YOLO('./best.pt')

save_dict = {0: "red", 1: "yellow", 2: 'green', 3: 'off'}
img_list = os.listdir('./images')
for imgs in img_list:
    img_path = f"./images/{imgs}"
    result = model.predict(img_path, save=True)
    for box in result[0].boxes:
        cls = int(box.cls[-1])
        direction = save_dict.get(cls, "").split('_')[-1]
        if direction == 'red':
            info = "前方是红灯请停车等候"
        elif direction == 'yellow':
            info = "前方是黄灯，马上要变为红灯，请减速慢行。⚠️"
        elif direction == 'green':
            info = "前方是绿灯可以保持匀速前进。"
        elif direction == 'off':
            info = "前方红绿灯已被关闭，请观察路面具体情况后前进"
        else:
            info = "很抱歉，展示没有成功识别红绿灯。"
        print(info)
