import os
import json
import glob
import numpy as np

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
RESULTS = f"{BASE}/results"
UNIFIED = f"{RESULTS}/unified"
os.makedirs(UNIFIED, exist_ok=True)

print("=" * 60)
print("步骤 1：处理已有的 Ensemble npy 文件")
print("=" * 60)

ens_files = ["ensemble_probs.npy", "ensemble_preds.npy", "ensemble_labels.npy"]
if all(os.path.exists(f"{RESULTS}/{f}") for f in ens_files):
    for f in ens_files:
        data = np.load(f"{RESULTS}/{f}")
        np.save(f"{UNIFIED}/{f}", data)
        print(f"  ✅ {f}: shape={data.shape}")
else:
    print("  ❌ Ensemble npy 文件不全")

print()
print("=" * 60)
print("步骤 2：转换所有 predictions_*.jsonl 文件")
print("=" * 60)

jsonl_files = glob.glob(f"{RESULTS}/predictions_*.jsonl")
print(f"找到 {len(jsonl_files)} 个 jsonl 文件\n")

for path in sorted(jsonl_files):
    fname = os.path.basename(path).replace("predictions_", "").replace(".jsonl", "")
    # 文件名里可能有冒号、空格、点，统一替换成下划线
    model_name = fname.replace(":", "_").replace(" ", "_").replace(".", "_")
    
    preds, labels = [], []
    fail = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("pred_label", -1) == -1:
                fail += 1
                continue
            preds.append(d["pred_label"])
            labels.append(d["true_label"])
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    np.save(f"{UNIFIED}/{model_name}_preds.npy", preds)
    np.save(f"{UNIFIED}/{model_name}_labels.npy", labels)
    print(f"  ✅ {model_name}")
    print(f"     preds: {len(preds)} 条 | labels: {len(labels)} 条 | 解析失败: {fail} 条")

print()
print("=" * 60)
print(f"统一格式已保存到: {UNIFIED}")
print("=" * 60)
print("\n生成的文件列表:")
for f in sorted(os.listdir(UNIFIED)):
    size = os.path.getsize(f"{UNIFIED}/{f}")
    print(f"  {f}  ({size:,} bytes)")