import os
import json
import numpy as np
from sklearn.metrics import f1_score

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
UNIFIED = f"{BASE}/results/unified"
OUTPUT = f"{BASE}/results/bootstrap_significance.json"

N_ITERATIONS = 10000
SEED = 42

def paired_bootstrap(preds_a, preds_b, labels, n_iter=N_ITERATIONS, seed=SEED):
    """对模型 A 和 B 做配对 bootstrap 显著性检验"""
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = []
    
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        f1_a = f1_score(labels[idx], preds_a[idx], average="macro")
        f1_b = f1_score(labels[idx], preds_b[idx], average="macro")
        deltas.append(f1_a - f1_b)
    
    deltas = np.array(deltas)
    mean_delta = float(deltas.mean())
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    
    # 双侧检验的 p 值（近似）
    if mean_delta >= 0:
        p_value = float((deltas <= 0).mean() * 2)
    else:
        p_value = float((deltas >= 0).mean() * 2)
    p_value = min(p_value, 1.0)
    
    significant = (ci_lower > 0) or (ci_upper < 0)
    return {
        "mean_delta": round(mean_delta, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value": round(p_value, 4),
        "significant": significant
    }

# ========== 加载所有模型 ==========
def load_model(name):
    preds = np.load(f"{UNIFIED}/{name}_preds.npy")
    labels = np.load(f"{UNIFIED}/{name}_labels.npy")
    return preds, labels

# 确定一个统一的 labels（都是 1200 条）
_, labels = load_model("bert_fgm")
print(f"测试集样本数: {len(labels)}\n")

# ========== 定义要对比的模型对 ==========
comparisons = [
    ("bert_fgm", "ensemble"),        # Ensemble 相比单模型
    ("bert_fgm", "roberta_fgm"),     # RoBERTa vs BERT
    ("bert_fgm", "qwen_lora"),       # 微调小模型 vs 本地 QLoRA
    ("ensemble", "qwen_lora"),       # 两个最强方案
    ("bert_fgm", "deepseek-flash_zero_shot"),  # 微调 vs LLM Zero-shot
]

results = {}

print("=" * 80)
print("配对 Bootstrap 显著性检验 (10000 次重采样)")
print("=" * 80)
print(f"{'对比':<40} {'ΔMacro-F1':>12} {'95% CI':>22} {'p值':>10} {'显著?':>8}")
print("-" * 95)

for model_a, model_b in comparisons:
    preds_a, _ = load_model(model_a)
    preds_b, _ = load_model(model_b)
    
    r = paired_bootstrap(preds_a, preds_b, labels)
    
    key = f"{model_a}_vs_{model_b}"
    results[key] = {
        "model_a": model_a,
        "model_b": model_b,
        "mean_delta": r["mean_delta"],
        "ci_95": r["ci_95"],
        "p_value": r["p_value"],
        "significant": r["significant"]
    }
    
    ci_str = f"[{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]"
    sig_str = "✅ 是" if r["significant"] else "❌ 否"
    
    print(f"{key:<40} {r['mean_delta']:>+12.4f} {ci_str:>22} {r['p_value']:>10.4f} {sig_str:>8}")

# ========== 保存 ==========
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print()
print("=" * 80)
print(f"结果已保存到: {OUTPUT}")
print("=" * 80)