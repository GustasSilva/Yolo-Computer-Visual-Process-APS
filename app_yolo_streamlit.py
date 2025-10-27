import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(page_title="YOLOv8 Detecta Webcam/Imagem", layout="wide")
st.title("🔍 Sistema de Detecção em Tempo Real - YOLOv11 + WebRTC + Imagem")

# Parâmetros e modelo
model_path = st.sidebar.text_input("Caminho do modelo (.pt):", "yolov8n.pt")
CONF_THRESHOLD = st.sidebar.slider("Confiança mínima", 0.1, 1.0, 0.35)
IOU_THRESHOLD = st.sidebar.slider("Limiar de IoU", 0.1, 1.0, 0.45)
IMG_SIZE = st.sidebar.select_slider("Tamanho da imagem (px)", [320, 480, 640, 768, 960], value=640)
track_objects = st.sidebar.checkbox("Usar tracker entre frames", value=True)

# Carregar modelo
@st.cache_resource
def load_model(path):
    model = YOLO(path)
    return model

model = load_model(model_path)

# Modo de operação
modo = st.sidebar.radio("Modo de operação:", ["📷 Webcam", "🖼️ Upload de imagem"])

if modo == "🖼️ Upload de imagem":
    uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Original", use_container_width=True)
        if st.button("Detectar Objetos na Imagem"):
            arr_img = np.array(img)
            results = model.predict(
                arr_img,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
            )
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="✅ Resultado YOLO", use_container_width=True)
            # Exibe objetos detectados
            if results[0].boxes:
                st.markdown("#### Objetos detectados:")
                counts = {}
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    counts[label] = counts.get(label, 0) + 1
                for k, v in counts.items():
                    st.write(f"- {k}: {v}")

elif modo == "📷 Webcam":
    st.markdown("#### Streaming contínuo da webcam com detecção em tempo real")
    def process_frame(frame):
        img = frame.to_ndarray(format="bgr24")
        if track_objects:
            results = model.track(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, tracker="bytetrack.yaml")
        else:
            results = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="yolo-stream",
        video_frame_callback=process_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

st.sidebar.info("Escolha entre streaming da webcam ou upload de imagem. A detecção pode ser ajustada com os parâmetros.")
