import os
from pathlib import Path

# Defina suas classes na ordem correta
CLASSES = [ 
    'alicate', 'capacete', 'chave_de_boca', 'chave_de_fenda', 'chave_de_grifo', 'chave_de_roda', 
    'chave_estrela', 'chave_inglesa', 'chave_phillips', 'esmerilhadeira', 'esquadro', 'estilete',
    'fita_isolante', 'furadeira', 'luva', 'martelo_de_borracha', 'martelo', 'nivel', 'oculos_de_protecao',
    'paquimetro', 'parafusadeira', 'regua', 'serra_circular', 'serrote', 'trena'
]

def extract_class_name_from_filename(filename):
    """
    Extrai o nome da classe do nome do arquivo.
    Exemplo: 'martelo1.txt' -> 'martelo'
              'chave_de_boca5.txt' -> 'chave_de_boca'
    """
    # Remove a extensão .txt
    name_without_ext = filename.replace('.txt', '')
    
    # Remove números do final
    class_name = ''.join([c for c in name_without_ext if not c.isdigit()])
    
    return class_name

def get_class_id(class_name, classes_list):
    """
    Retorna o ID da classe baseado na lista de classes.
    """
    try:
        return classes_list.index(class_name)
    except ValueError:
        print(f"⚠️ Aviso: Classe '{class_name}' não encontrada na lista de classes!")
        return None

def process_label_file(label_path, classes_list):
    """
    Processa um arquivo de label e corrige o ID da classe.
    """
    filename = os.path.basename(label_path)
    class_name = extract_class_name_from_filename(filename)
    correct_class_id = get_class_id(class_name, classes_list)
    
    if correct_class_id is None:
        return False
    
    # Lê o conteúdo do arquivo
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Processa cada linha
        corrected_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Ignora linhas vazias
                parts = line.split()
                if len(parts) >= 5:
                    # Substitui o ID da classe pelo correto
                    parts[0] = str(correct_class_id)
                    corrected_lines.append(' '.join(parts))
        
        # Escreve de volta no arquivo
        with open(label_path, 'w') as f:
            for line in corrected_lines:
                f.write(line + '\n')
        
        return True
    
    except Exception as e:
        print(f"❌ Erro ao processar {label_path}: {e}")
        return False

def process_dataset(dataset_path, classes_list):
    """
    Processa apenas os diretórios train e valid.
    """
    dataset_path = Path(dataset_path)
    
    # Diretórios a processar - APENAS train e valid
    directories = [
        dataset_path / 'alldataset' / 'labels',
        dataset_path / 'train' / 'labels',
        dataset_path / 'valid' / 'labels'
    ]
    
    total_processed = 0
    total_errors = 0
    
    for label_dir in directories:
        if not label_dir.exists():
            print(f"⚠️ Diretório não encontrado: {label_dir}")
            continue
        
        print(f"\n📂 Processando: {label_dir}")
        
        # Processa todos os arquivos .txt no diretório
        label_files = list(label_dir.glob('*.txt'))
        
        print(f"   📄 Encontrados {len(label_files)} arquivos .txt")
        
        for label_file in label_files:
            if process_label_file(label_file, classes_list):
                total_processed += 1
                print(f"   ✅ {label_file.name}")
            else:
                total_errors += 1
                print(f"   ❌ {label_file.name}")
    
    print(f"\n{'='*60}")
    print(f"📊 Resumo Final:")
    print(f"   ✅ Arquivos processados com sucesso: {total_processed}")
    print(f"   ❌ Erros: {total_errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Defina o caminho para o seu dataset
    DATASET_PATH = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\dataset"  # Ajuste conforme necessário
    
    print("🚀 Iniciando reorganização dos labels do YOLO...")
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"🏷️  Total de classes: {len(CLASSES)}")
    print("\n⚠️  ATENÇÃO: Processando APENAS train/ e valid/")
    print("   (alldataset/ será ignorado)\n")
    
    process_dataset(DATASET_PATH, CLASSES)
    
    print("\n✨ Processo concluído!")
