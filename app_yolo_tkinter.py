import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# Parâmetros globais
MODEL_PATH = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\models\best.pt"
model = YOLO(MODEL_PATH)
IMG_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Tamanho exato da área de exibição
DISPLAY_W, DISPLAY_H = 800, 600

root = tk.Tk()
root.title("🔍 YOLOv8 - Webcam & Imagem")
root.configure(bg="#f6f6f6")

# Define tamanho da janela (um pouco maior que o canvas)
root.geometry(f"{DISPLAY_W+40}x{DISPLAY_H+260}")
root.resizable(False, False)

# Título
title = tk.Label(root, text="Detecção de Objetos YOLOv11", font=("Arial", 20, "bold"), bg="#f6f6f6")
title.pack(pady=(20, 5))

# Frame central para imagem/vídeo
frame_canvas = tk.Frame(root, bg="#f6f6f6")
frame_canvas.pack()

canvas = tk.Canvas(frame_canvas, width=DISPLAY_W, height=DISPLAY_H, bg="#dedede", highlightthickness=1, highlightbackground="#ccc")
canvas.pack()

# Botões de operação
frame_ops = tk.Frame(root, bg="#f6f6f6")
frame_ops.pack(pady=14)

btn_upload = tk.Button(frame_ops, text="Upload de Imagem", font=("Arial", 14), width=18, bg="#2f90d3", fg="white")
btn_upload.pack(side=tk.LEFT, padx=12)

btn_webcam = tk.Button(frame_ops, text="Iniciar Webcam", font=("Arial", 14), width=18, bg="#309e6b", fg="white")
btn_webcam.pack(side=tk.LEFT, padx=12)

btn_stop = tk.Button(frame_ops, text="Parar Webcam", font=("Arial", 14), width=18, bg="#d95858", fg="white")
btn_stop.pack(side=tk.LEFT, padx=12)

# Informações de detecção
frame_info = tk.Frame(root, bg="#f6f6f6")
frame_info.pack(pady=10)

label_info = tk.Label(frame_info, text="Objetos Detectados:", font=("Arial", 14, "bold"), bg="#f6f6f6")
label_info.pack()

info_box = tk.Text(frame_info, height=6, width=40, font=("Arial", 12), bg="#ededed", relief=tk.FLAT, padx=10, pady=8)
info_box.pack(pady=2)
info_box.tag_configure('center', justify='center')

cap = None
streaming = False

def detect_and_show(img_arr):
    results = model.predict(
        img_arr,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE
    )
    annotated = results[0].plot()
    return annotated, results[0].boxes

def upload_img():
    global streaming
    streaming = False
    file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
    if file_path:
        img = cv2.imread(file_path)
        # img já está BGR do OpenCV
        annotated, boxes = detect_and_show(img)
        # Converta apenas uma vez para RGB
        im_display = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        im_display = im_display.resize((DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(im_display)
        canvas.imgtk = imgtk
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=imgtk)
        update_box_info(boxes)


def update_box_info(boxes):
    info_box.delete("1.0", tk.END)
    if boxes and len(boxes) > 0:
        counts = {}
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            counts[label] = counts.get(label, 0) + 1
        text = "\n".join([f"{k}: {v}" for k, v in counts.items()])
        info_box.insert(tk.END, text, 'center')
    else:
        info_box.insert(tk.END, "Nenhum objeto detectado.", 'center')

def start_webcam():
    global cap, streaming
    streaming = True
    cap = cv2.VideoCapture(0)
    show_webcam()

def stop_webcam():
    global cap, streaming
    streaming = False
    if cap:
        cap.release()
        cap = None

def show_webcam():
    global cap, streaming
    if streaming and cap is not None:
        ret, frame = cap.read()
        if ret:
            # Detecção: envie BGR para YOLO, mas converta só uma vez para Pillow
            annotated, boxes = detect_and_show(frame)
            # Converta uma única vez para RGB para exibição, nunca mais
            im_display = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            im_display = im_display.resize((DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(im_display)
            canvas.imgtk = imgtk
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=imgtk)
            update_box_info(boxes)
        canvas.after(30, show_webcam)


btn_upload.config(command=upload_img)
btn_webcam.config(command=start_webcam)
btn_stop.config(command=stop_webcam)

root.protocol("WM_DELETE_WINDOW", lambda: (stop_webcam(), root.destroy()))
root.mainloop()
