import os
import json
import numpy as np

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
RESULTS = f"{BASE}/results"

# ========== 1. 加载 Zero-shot 和 Few-shot 的预测 ==========
def load_predictions(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds[d["id"]] = {
                "text": d["text"],
                "true_label": d["true_label"],
                "pred_label": d["pred_label"]
            }
    return preds

zs = load_predictions(f"{RESULTS}/predictions_deepseek-flash_zero_shot.jsonl")
fs = load_predictions(f"{RESULTS}/predictions_deepseek-flash_few_shot.jsonl")

print(f"Zero-shot 样本: {len(zs)}")
print(f"Few-shot 样本: {len(fs)}\n")

# ========== 2. 加载 BERT 的置信度（找难例） ==========
confidence = np.load(f"{RESULTS}/unified/ensemble_confidence.npy") \
    if os.path.exists(f"{RESULTS}/unified/ensemble_confidence.npy") \
    else np.load(f"{RESULTS}/ensemble_confidence.npy")
all_labels = np.load(f"{RESULTS}/unified/test_labels.npy")

# ========== 3. 定义难例子集 ==========
# 难例 = BERT+Ensemble 置信度最低的 20%（约 240 条）
threshold = np.percentile(confidence, 20)
hard_indices = set(np.where(confidence <= threshold)[0].tolist())
print(f"难例阈值（置信度 20% 分位）: {threshold:.4f}")
print(f"难例子集大小: {len(hard_indices)}\n")

# ========== 4. 对比 Zero-shot vs Few-shot 在三个子集上的表现 ==========
def evaluate_on_subset(subset_indices, name):
    zs_ok, fs_ok = 0, 0
    zs_total, fs_total = 0, 0
    zs_only_correct = 0
    fs_only_correct = 0
    both_correct = 0
    both_wrong = 0
    
    for idx in subset_indices:
        if idx not in zs or idx not in fs:
            continue
        true = zs[idx]["true_label"]
        zs_pred = zs[idx]["pred_label"]
        fs_pred = fs[idx]["pred_label"]
        
        if zs_pred == -1 or fs_pred == -1:
            continue
        
        zs_correct = (zs_pred == true)
        fs_correct = (fs_pred == true)
        
        zs_total += 1
        fs_total += 1
        if zs_correct: zs_ok += 1
        if fs_correct: fs_ok += 1
        
        if zs_correct and fs_correct: both_correct += 1
        elif (not zs_correct) and fs_correct: fs_only_correct += 1
        elif zs_correct and (not fs_correct): zs_only_correct += 1
        else: both_wrong += 1
    
    zs_acc = zs_ok / zs_total if zs_total > 0 else 0
    fs_acc = fs_ok / fs_total if fs_total > 0 else 0
    
    print(f"========== {name} (n={zs_total}) ==========")
    print(f"  Zero-shot 准确率: {zs_acc:.4f}")
    print(f"  Few-shot 准确率:  {fs_acc:.4f}")
    print(f"  提升: {(fs_acc - zs_acc) * 100:+.2f} 个百分点")
    print(f"  ├─ 两者都对: {both_correct}")
    print(f"  ├─ 两者都错: {both_wrong}")
    print(f"  ├─ 仅 Zero-shot 对: {zs_only_correct}")
    print(f"  └─ 仅 Few-shot 对: {fs_only_correct}")
    print()
    
    return {
        "n": zs_total,
        "zero_shot_acc": round(zs_acc, 4),
        "few_shot_acc": round(fs_acc, 4),
        "delta": round(fs_acc - zs_acc, 4),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "zs_only_correct": zs_only_correct,
        "fs_only_correct": fs_only_correct
    }

# 全量
all_indices = set(range(1200))
r_all = evaluate_on_subset(all_indices, "全量测试集")

# 难例 20%
r_hard = evaluate_on_subset(hard_indices, "难例子集（置信度最低 20%）")

# 简单样本（BERT 置信度最高的 50%）
easy_threshold = np.percentile(confidence, 50)
easy_indices = set(np.where(confidence > easy_threshold)[0].tolist())
r_easy = evaluate_on_subset(easy_indices, "简单样本（置信度最高 50%）")

# ========== 5. 保存 ==========
output = {
    "hard_subset_threshold": round(float(threshold), 4),
    "all": r_all,
    "hard": r_hard,
    "easy": r_easy
}
with open(f"{RESULTS}/fewshot_hard_subset_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("=" * 60)
print(f"结果已保存到: {RESULTS}/fewshot_hard_subset_analysis.json")
print("=" * 60)