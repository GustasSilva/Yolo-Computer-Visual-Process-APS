from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time

# ==============================
# CONFIGURAÇÕES INICIAIS
# ==============================
model_path = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\models\best.pt"

# Carrega o modelo YOLO treinado (usa GPU se disponível)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
model = YOLO(model_path)
model.to(device)

# ==============================
# CONFIGURAÇÕES DE CÂMERA
# ==============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise IOError("❌ Não foi possível acessar a webcam.")

print("✅ Webcam iniciada. Pressione 'q' para encerrar.")

# ==============================
# AJUSTES DE DETECÇÃO
# ==============================
CONF_THRESHOLD = 0.35   # confiança mínima
IOU_THRESHOLD = 0.45    # limiar de sobreposição
FRAME_SKIP = 1          # processa 1 a cada N frames (ajuste se quiser mais velocidade)
DELAY = 1               # tempo entre frames

# ==============================
# LOOP PRINCIPAL
# ==============================
frame_count = 0
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Falha ao capturar frame.")
        break

    frame_count += 1

    # Reduz brilho excessivo (melhora detecção de fundo claro)
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=-30)

    # Processa apenas 1 de cada N frames
    if frame_count % FRAME_SKIP == 0:
        # Reduz ruído e melhora contraste
        frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        frame_yuv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YUV)
        frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
        frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

        # Inferência YOLO
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=device,
            imgsz=768  # pode aumentar para 960 se quiser mais precisão
        )

        # Renderiza detecções
        annotated_frame = results[0].plot()

        # Cálculo de FPS
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()

        # Mostra FPS e dispositivo
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} ({device.upper()})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Exibe frame
        cv2.imshow("🔍 Detecção em Tempo Real - YOLO", annotated_frame)

    # Sai ao pressionar 'q'
    if cv2.waitKey(DELAY) & 0xFF == ord('q'):
        break

# ==============================
# FINALIZAÇÃO
# ==============================
cap.release()
cv2.destroyAllWindows()
print("🟢 Execução finalizada com sucesso.")
