import os
import random
import shutil

# === CONFIGURAÇÕES PRINCIPAIS ===
base_dir = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\dataset"

orig_images_dir = os.path.join(base_dir, "alldataset", "images")
orig_labels_dir = os.path.join(base_dir, "alldataset", "labels")

train_img_dir = os.path.join(base_dir, "train", "images")
valid_img_dir = os.path.join(base_dir, "valid", "images")
train_lbl_dir = os.path.join(base_dir, "train", "labels")
valid_lbl_dir = os.path.join(base_dir, "valid", "labels")

# Cria pastas caso não existam
for d in [train_img_dir, valid_img_dir, train_lbl_dir, valid_lbl_dir]:
    os.makedirs(d, exist_ok=True)


# === FUNÇÕES AUXILIARES ===
def get_classes_from_label(label_path):
    """Lê as classes de um arquivo .txt YOLO"""
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r") as f:
        lines = f.readlines()
    return [int(line.split()[0]) for line in lines if line.strip()]


def copy_pair(image_name, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """Copia imagem e label correspondente"""
    base_name, _ = os.path.splitext(image_name)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        src_img_path = os.path.join(src_img_dir, base_name + ext)
        if os.path.exists(src_img_path):
            break
    else:
        return False  # não achou imagem correspondente

    src_lbl_path = os.path.join(src_lbl_dir, base_name + ".txt")

    shutil.copy(src_img_path, dst_img_dir)
    if os.path.exists(src_lbl_path):
        shutil.copy(src_lbl_path, dst_lbl_dir)
    return True


# === COLETA DE LABELS E CLASSES ===
label_files = [f for f in os.listdir(orig_labels_dir) if f.endswith(".txt")]
class_to_files = {}

for lbl in label_files:
    classes = get_classes_from_label(os.path.join(orig_labels_dir, lbl))
    for c in classes:
        class_to_files.setdefault(c, []).append(lbl)

# === SEPARAÇÃO DE DADOS ===
val_files = set()

# 1️⃣ Garante pelo menos 1 exemplo de cada classe no conjunto de validação
for c, files in class_to_files.items():
    val_files.add(random.choice(files))

# 2️⃣ Garante que TODAS as classes realmente estejam representadas
def get_classes_present_in_files(files_set):
    present = set()
    for f in files_set:
        present.update(get_classes_from_label(os.path.join(orig_labels_dir, f)))
    return present

present_classes = get_classes_present_in_files(val_files)
missing_classes = set(class_to_files.keys()) - present_classes

for c in missing_classes:
    val_files.add(random.choice(class_to_files[c]))

# 3️⃣ Pega o restante das imagens e divide 80/20
remaining_labels = list(set(label_files) - val_files)
random.shuffle(remaining_labels)
extra_val = int(0.1 * len(remaining_labels))
val_files.update(remaining_labels[:extra_val])
train_files = list(set(label_files) - val_files)

# === COPIA OS ARQUIVOS ===
for lbl_file in train_files:
    copy_pair(lbl_file.replace(".txt", ".jpg"), orig_images_dir, orig_labels_dir, train_img_dir, train_lbl_dir)

for lbl_file in val_files:
    copy_pair(lbl_file.replace(".txt", ".jpg"), orig_images_dir, orig_labels_dir, valid_img_dir, valid_lbl_dir)

# === RELATÓRIO FINAL ===
print("✅ Dataset dividido com sucesso!")
print(f"Treino: {len(train_files)} imagens")
print(f"Validação: {len(val_files)} imagens")
print(f"Classes totais encontradas: {len(class_to_files)}")
print(f"Classes presentes no conjunto de validação: {len(get_classes_present_in_files(val_files))}")
