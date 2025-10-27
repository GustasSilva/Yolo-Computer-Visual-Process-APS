import os
from collections import Counter
labels_dir = r"C:\Users\Gustavo\Desktop\APS YOLO 1.0\dataset\alldataset\labels"
cnt = Counter()
for f in os.listdir(labels_dir):
    if not f.endswith(".txt"): continue
    with open(os.path.join(labels_dir,f)) as fh:
        for l in fh:
            s = l.strip().split()
            if not s: continue
            cnt[int(s[0])] += 1
print("Total arquivos de label:", len([f for f in os.listdir(labels_dir) if f.endswith(".txt")]))
for k,v in sorted(cnt.items()):
    print(f"class {k}: {v} instâncias")
