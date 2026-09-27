import os
import json
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    classification_report
)

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
UNIFIED = f"{BASE}/results/unified"
OUTPUT = f"{BASE}/results/unified_evaluation.json"

# ========== 1. 扫描所有 preds/labels 文件对 ==========
preds_files = [f for f in os.listdir(UNIFIED) if f.endswith("_preds.npy")]
models = sorted([f.replace("_preds.npy", "") for f in preds_files])

print(f"发现 {len(models)} 个模型: {models}\n")

results = {}

for model_name in models:
    preds_path = f"{UNIFIED}/{model_name}_preds.npy"
    labels_path = f"{UNIFIED}/{model_name}_labels.npy"
    
    if not os.path.exists(labels_path):
        print(f"⚠️  {model_name}: 缺少 labels 文件，跳过")
        continue
    
    preds = np.load(preds_path)
    labels = np.load(labels_path)
    
    if len(preds) != len(labels):
        print(f"⚠️  {model_name}: preds 与 labels 长度不一致 ({len(preds)} vs {len(labels)})，跳过")
        continue
    
    # ========== 2. 处理解析失败（pred == -1） ==========
    valid_mask = preds != -1
    n_total = len(preds)
    n_fail = int((~valid_mask).sum())
    parse_fail_rate = n_fail / n_total
    
    if valid_mask.sum() == 0:
        print(f"⚠️  {model_name}: 全部解析失败，跳过")
        continue
    
    y_true = labels[valid_mask]
    y_pred = preds[valid_mask]
    
    # ========== 3. 计算所有指标 ==========
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    
    neg_f1 = f1_score(y_true, y_pred, pos_label=0, average="binary")
    pos_f1 = f1_score(y_true, y_pred, pos_label=1, average="binary")
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # ========== 4. 保存结果 ==========
    results[model_name] = {
        "n_total": n_total,
        "n_valid": int(valid_mask.sum()),
        "n_fail": n_fail,
        "parse_fail_rate": round(parse_fail_rate, 4),
        "macro_f1": round(float(macro_f1), 4),
        "accuracy": round(float(acc), 4),
        "neg_f1": round(float(neg_f1), 4),
        "pos_f1": round(float(pos_f1), 4),
        "neg_recall": round(float(neg_recall), 4),
        "pos_recall": round(float(pos_recall), 4),
        "neg_precision": round(float(neg_precision), 4),
        "pos_precision": round(float(pos_precision), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }
    
    print(f"✅ {model_name}")
    print(f"   Macro-F1={macro_f1:.4f} | Acc={acc:.4f} | 解析失败率={parse_fail_rate*100:.2f}%")
    print(f"   负面 F1={neg_f1:.4f} | 正面 F1={pos_f1:.4f}")
    print(f"   负面 Recall={neg_recall:.4f} | 正面 Recall={pos_recall:.4f}")
    print(f"   混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

# ========== 5. 保存汇总 JSON ==========
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("=" * 70)
print(f"汇总已保存到: {OUTPUT}")

# ========== 6. 打印对比表格 ==========
print("=" * 70)
print("\n========== 全模型统一对比表 ==========")
header = f"{'模型':<30} {'Macro-F1':>10} {'负面F1':>8} {'正面F1':>8} {'负面R':>8} {'正面R':>8} {'失败率':>8}"
print(header)
print("-" * len(header))
for name, m in results.items():
    print(f"{name:<30} {m['macro_f1']:>10.4f} {m['neg_f1']:>8.4f} {m['pos_f1']:>8.4f} "
          f"{m['neg_recall']:>8.4f} {m['pos_recall']:>8.4f} {m['parse_fail_rate']*100:>7.2f}%")