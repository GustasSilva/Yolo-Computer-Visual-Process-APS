import os
from ultralytics import YOLO

# Caminhos padrão (ajuste conforme sua máquina)
DATA_YAML = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\dataset\data.yaml"  # Caminho para o arquivo YAML do dataset
IMAGEM_TESTE = r"C:\Users\Gustavo\runs\detect\predict\chave de fenda.jpg"
PESO_TREINADO = r"C:\Users\Gustavo\runs\detect\train6\weights\best.pt"

def treinar(modelo_base="yolov8m.pt", epochs=50, imgsz=640):
    print(f"\n🚀 Iniciando treino com {modelo_base} por {epochs} epochs...\n")
    model = YOLO(modelo_base)
    model.train(data=DATA_YAML, epochs=epochs, imgsz=imgsz, project="runs/detect", name="train_auto")
    print("\n✅ Treino concluído! Pesos salvos em runs/detect/train_auto/weights/\n")

def validar(peso_modelo=PESO_TREINADO):
    print(f"\n🔎 Validando modelo: {peso_modelo}\n")
    model = YOLO(peso_modelo)
    model.val(data=DATA_YAML)
    print("\n✅ Validação concluída. Resultados em runs/detect/train_auto/\n")

def predizer(peso_modelo=PESO_TREINADO, imagem=IMAGEM_TESTE):
    print(f"\n📷 Realizando predição em {imagem}\n")
    model = YOLO(peso_modelo)
    results = model.predict(source=imagem, show=True, save=True, conf=0.25)
    print(f"\n✅ Predição concluída! Resultados salvos em runs/detect/predict/\n")

def menu():
    while True:
        print("""
==========================
 YOLOv8 - Menu de Ações
==========================
1️⃣  Treinar modelo
2️⃣  Validar modelo
3️⃣  Fazer predição em imagem
4️⃣  Sair
""")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            modelo = input("Modelo base (ex: yolov8n.pt, yolov8s.pt, yolov8m.pt): ") or "yolov8m.pt"
            epocas = int(input("Quantas epochs deseja treinar? (padrão 50): ") or 50)
            treinar(modelo, epocas)
        elif opcao == "2":
            peso = input("Caminho do modelo treinado (.pt): ") or PESO_TREINADO
            validar(peso)
        elif opcao == "3":
            peso = input("Caminho do modelo (.pt): ") or PESO_TREINADO
            imagem = input("Caminho da imagem para teste: ") or IMAGEM_TESTE
            predizer(peso, imagem)
        elif opcao == "4":
            print("\n👋 Saindo do sistema YOLO Tool...\n")
            break
        else:
            print("❌ Opção inválida! Tente novamente.\n")

if __name__ == "__main__":
    menu()
