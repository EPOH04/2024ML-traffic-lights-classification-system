from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import inspect
import textwrap
import streamlit as st

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image, conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)
                return res

def example_img(conf, model):
    source_img = './example.jpg'
    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )


    with st.spinner("Running..."):
        res = model.predict(uploaded_image, conf=conf)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]

        with col2:
            st.image(res_plotted,
                     caption="Detected Image",
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("No image is uploaded yet!")
                st.write(ex)
        return res


st.set_page_config(page_title="红绿灯检测", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.sidebar.header("来开始配置模型吧~")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = "success_seed"
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        ['Yolo_traffic',]
    )
else:
    st.error("请选择正确的模型")

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) / 100

model_path = r"./best.pt"

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"模型加载失败，请检查您的路径{model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Image"]
)

source_img = None

save_dict = {0: "red", 1: "yellow", 2: 'green', 3: 'off'}
if source_selectbox == "Image": # Image
    result = infer_uploaded_image(confidence, model)
    if result:
        for box in result[0].boxes:
            cls = int(box.cls[-1])
            direction = save_dict.get(cls, "").split('_')[-1]
            if direction == 'red':
                info = "前方是红灯请停车等候"
                st.error(info)
            elif direction == 'yellow':
                info = "前方是黄灯，马上要变为红灯，请减速慢行。⚠️"
                st.error(info)
            elif direction == 'green':
                info = "前方是绿灯可以保持匀速前进。"
                st.success(info)
            elif direction == 'off':
                info = "前方红绿灯已被关闭，请观察路面具体情况后前进"
                st.error(info)
            else:
                info = "很抱歉，没有成功识别红绿灯。"
                st.info(info)
else:
    st.error("只能选择图片哦")

if st.sidebar.button("样例"):
    result = example_img(confidence, model)
    if result:
        for box in result[0].boxes:
            cls = int(box.cls[-1])
            direction = save_dict.get(cls, "").split('_')[-1]
            if direction == 'red':
                info = "前方是红灯请停车等候"
                st.error(info)
            elif direction == 'yellow':
                info = "前方是黄灯，马上要变为红灯，请减速慢行。⚠️"
                st.error(info)
            elif direction == 'green':
                info = "前方是绿灯可以保持匀速前进。"
                st.success(info)
            elif direction == 'off':
                info = "前方红绿灯已被关闭，请观察路面具体情况后前进"
                st.error(info)
            else:
                info = "很抱歉，没有成功识别红绿灯。"
                st.info(info)

