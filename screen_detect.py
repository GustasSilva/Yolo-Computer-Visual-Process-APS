"""
screen_detect.py

Captura a tela (ou região) e roda detecções em tempo real com um modelo YOLO (Ultralytics).

Teclas:
  q - sair
  s - salvar screenshot anotado (pasta ./screens)
  t - alternar exibição rótulos
  p - alternar pausar/resumir inferência
  +/- - aumentar/diminuir confiança mínima (conf threshold)

Configurações rápidas no topo do script.
"""

import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mss

# -----------------------------
# Configurações (edite se quiser)
# -----------------------------
MODEL_PATH = r"C:\Users\Gustavo\runs\detect\train36\weights\best.pt"  # caminho do seu .pt
CONF_THRESHOLD_DEFAULT = 0.35
IOU_THRESHOLD_DEFAULT = 0.45
USE_GPU_IF_AVAILABLE = True
IMGSZ = 960  # tamanho com que o modelo receberá o frame (pode reduzir para 640 se lenta)
SAVE_DIR = "screens"
IGNORE_BOX_AREA_MAX = 0.85  # ignora caixas que ocupam > 85% do frame (ajuste)
CAPTURE_REGION = None  # None para tela inteira, ou dict {'top':..., 'left':..., 'width':..., 'height':...}
# -----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD_DEFAULT)
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD_DEFAULT)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--region", type=str, default=None,
                   help="Região de captura como left,top,width,height (ex: 0,0,800,600) ou 'all' para tela inteira")
    p.add_argument("--mon", type=int, default=1, help="Monitor index (mss monitors list index)")
    return p.parse_args()

def screen_to_bgr(sct_img):
    # mss retorna BGRA ou raw bytes; convert to BGR numpy
    img = np.array(sct_img)
    # mss returns BGRA on some platforms; check shape
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def draw_boxes(frame, boxes, scores, classes, names, conf_th):
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
        # normalizar e filtrar por confiança
        if conf < conf_th:
            continue
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        area_frac = ((x2i - x1i) * (y2i - y1i)) / (w * h)
        if area_frac > IGNORE_BOX_AREA_MAX:
            # ignora caixas muito grandes (ex: parede dominando)
            continue
        label = f"{names[int(cls)]} {conf:.2f}" if names else f"{int(cls)} {conf:.2f}"
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
        # fundo para texto
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (x1i, y1i - tsize[1] - 6), (x1i + tsize[0] + 6, y1i), color, -1)
        cv2.putText(frame, label, (x1i + 3, y1i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def main():
    args = parse_args()

    device = "cuda" if (torch.cuda.is_available() and USE_GPU_IF_AVAILABLE) else "cpu"
    print(f"[INFO] Usando dispositivo: {device}")

    print("[INFO] Carregando modelo:", args.model)
    model = YOLO(args.model)
    # Not always necessary to call .to(device) with ultralytics high-level API, but do it for safety:
    try:
        model.to(device)
    except Exception:
        pass

    # nomes das classes (se disponível)
    try:
        names = model.model.names  # ultralytics model names
    except Exception:
        names = None

    conf_th = args.conf
    iou_th = args.iou
    imgsz = args.imgsz

    # configurar captura de tela com mss
    sct = mss.mss()
    monitors = sct.monitors  # list
    mon_idx = args.mon if args.mon < len(monitors) else 1
    monitor = monitors[mon_idx]  # default monitor
    region = None
    if args.region and args.region.lower() != "all":
        try:
            l, t, w, h = map(int, args.region.split(","))
            region = {"left": l, "top": t, "width": w, "height": h}
        except Exception:
            print("[WARN] region inválida, usando monitor inteiro.")
            region = monitor
    else:
        region = monitor if args.region != "all" else monitor

    print(f"[INFO] Capturando região: {region}")

    paused = False
    show_labels = True
    save_count = 0
    last_time = time.time()
    fps = 0.0

    print("Pressione 'q' para sair, 's' para salvar screenshot, 't' alternar rótulos, 'p' pausar.")

    while True:
        start = time.time()
        sct_img = sct.grab(region)
        frame = screen_to_bgr(sct_img)

        # opcional pré-processamento (aumenta robustez)
        frame_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not paused:
            # inference - we can pass np array directly
            results = model.predict(source=frame_proc,
                                    conf=conf_th,
                                    iou=iou_th,
                                    device=device,
                                    imgsz=imgsz,
                                    verbose=False)
            r = results[0]

            # extrair boxes, scores, classes
            try:
                boxes_tensor = r.boxes.xyxy.cpu().numpy()  # (N,4)
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
            except Exception:
                # fallback: use .boxes.data
                data = r.boxes.data.cpu().numpy()
                if data.size == 0:
                    boxes_tensor = np.array([])
                    scores = np.array([])
                    classes = np.array([])
                else:
                    boxes_tensor = data[:, :4]
                    scores = data[:, 4]
                    classes = data[:, 5]

            # plot manual (evita sobrescrita do plot padrão)
            disp_frame = frame.copy()
            disp_frame = draw_boxes(disp_frame, boxes_tensor, scores, classes, names, conf_th) if show_labels else disp_frame

            # info overlay
            end = time.time()
            elapsed = end - last_time
            fps = 0.9 * fps + 0.1 * (1.0 / (end - start)) if fps else (1.0 / (end - start))
            last_time = end
            cv2.putText(disp_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(disp_frame, f"Conf:{conf_th:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Screen YOLO Detect", disp_frame)
        else:
            # mostrar frame pausado
            cv2.putText(frame, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow("Screen YOLO Detect", frame)

        # teclado
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            # salvar screenshot anotado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"screen_{timestamp}.jpg")
            cv2.imwrite(out_path, disp_frame if not paused else frame)
            print(f"[SAVE] {out_path}")
        elif k == ord("t"):
            show_labels = not show_labels
            print("[INFO] Mostrar labels:", show_labels)
        elif k == ord("p"):
            paused = not paused
            print("[INFO] Paused:", paused)
        elif k == ord("+"):
            conf_th = min(0.99, conf_th + 0.05)
            print("[INFO] conf:", conf_th)
        elif k == ord("-"):
            conf_th = max(0.01, conf_th - 0.05)
            print("[INFO] conf:", conf_th)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
