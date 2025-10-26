import os

# Caminho da pasta labels (ajuste se necessário)
labels_dir = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\dataset\valid\labels"

# Classes do YAML
yaml_classes = [
    'alicate', 'chave_de_boca', 'chave_de_fenda', 'chave_de_grifo', 'chave_de_roda',
    'chave_estrela', 'chave_inglesa', 'chave_phillips', 'esmerilhadeira', 'esquadro',
    'estilete', 'furadeira', 'martelo_de_borracha', 'martelo', 'nivel',
    'oculos_de_protecao', 'paquimetro', 'parafusadeira', 'regua', 'serra_circular',
    'serrote', 'trena'
]

found_classes = set()

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        with open(os.path.join(labels_dir, file), "r") as f:
            for line in f:
                cls_id = int(line.split()[0])
                found_classes.add(cls_id)

missing = [yaml_classes[i] for i in range(len(yaml_classes)) if i not in found_classes]

print(f"Classes encontradas: {len(found_classes)} / {len(yaml_classes)}")
print("Ausentes:", missing)
