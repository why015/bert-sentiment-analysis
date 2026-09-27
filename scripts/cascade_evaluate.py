import os
import json
import time
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ========== 配置 ==========
BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
# ==========================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========== 1. 加载 Ensemble 全量预测 ==========
all_probs = np.load(f"{BASE}/results/ensemble_probs.npy")
all_preds = np.load(f"{BASE}/results/ensemble_preds.npy")
all_labels = np.load(f"{BASE}/results/ensemble_labels.npy")
confidence = np.load(f"{BASE}/results/ensemble_confidence.npy")

with open(f"{BASE}/results/hard_samples_5pct.json", "r", encoding="utf-8") as f:
    hard_samples = json.load(f)

print(f"全量样本: {len(all_labels)}")
print(f"难例数: {len(hard_samples)}")

# ========== 2. 调 DeepSeek 重判难例 ==========
PROMPT_TEMPLATE = """请判断以下评论的情感是正面还是负面。只输出"正面"或"负面"，不要输出任何其他内容。

评论：{text}"""

def parse_output(text):
    if "正面" in text and "负面" not in text:
        return 1
    if "负面" in text and "正面" not in text:
        return 0
    pos_idx = text.rfind("正面")
    neg_idx = text.rfind("负面")
    if pos_idx > neg_idx:
        return 1
    if neg_idx > pos_idx:
        return 0
    return -1

llm_preds = []
llm_latencies = []
llm_input_tokens = 0
llm_output_tokens = 0
fail_count = 0

print("\n开始调用 DeepSeek 重判难例...")
for sample in tqdm(hard_samples):
    prompt = PROMPT_TEMPLATE.format(text=sample["text"])
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )
        latency = time.time() - start
        llm_latencies.append(latency)
        
        raw = response.choices[0].message.content.strip()
        pred = parse_output(raw)
        llm_input_tokens += response.usage.prompt_tokens
        llm_output_tokens += response.usage.completion_tokens
        llm_preds.append(pred)
        
        sample["llm_pred"] = pred
        sample["llm_raw"] = raw
        sample["llm_latency"] = latency
    except Exception as e:
        print(f"\n[Error] idx={sample['idx']}: {e}")
        llm_preds.append(-1)
        sample["llm_pred"] = -1
        fail_count += 1

# ========== 3. 保存 DeepSeek 难例预测 ==========
with open(f"{BASE}/results/cascade_llm_predictions.json", "w", encoding="utf-8") as f:
    json.dump(hard_samples, f, ensure_ascii=False, indent=2)

# ========== 4. 融合：难例用 LLM 结果，其余用 Ensemble ==========
cascade_preds = all_preds.copy()
for sample in hard_samples:
    if sample["llm_pred"] != -1:
        cascade_preds[sample["idx"]] = sample["llm_pred"]

# ========== 5. 对比指标 ==========
def compute_metrics(labels, preds, name):
    valid = preds != -1
    y_true, y_pred = labels[valid], preds[valid]
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    neg_f1 = f1_score(y_true, y_pred, pos_label=0, average="binary")
    pos_f1 = f1_score(y_true, y_pred, pos_label=1, average="binary")
    neg_recall = (y_pred[y_true == 0] == 0).mean()
    pos_recall = (y_pred[y_true == 1] == 1).mean()
    print(f"\n========== {name} ==========")
    print(f"Macro-F1: {macro_f1:.4f} | Accuracy: {acc:.4f}")
    print(f"负面 F1: {neg_f1:.4f} | 正面 F1: {pos_f1:.4f}")
    print(f"负面 Recall: {neg_recall:.4f} | 正面 Recall: {pos_recall:.4f}")
    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "neg_f1": round(neg_f1, 4),
        "pos_f1": round(pos_f1, 4),
        "neg_recall": round(neg_recall, 4),
        "pos_recall": round(pos_recall, 4)
    }

baseline_metrics = compute_metrics(all_labels, all_preds, "Baseline: BERT+Ensemble 全量")
cascade_metrics = compute_metrics(all_labels, cascade_preds, "Cascade: Ensemble + LLM 难例兜底")

# 只看难例 60 条上的效果
hard_labels = all_labels[[s["idx"] for s in hard_samples]]
hard_ens_preds = all_preds[[s["idx"] for s in hard_samples]]
hard_llm_preds = np.array(llm_preds)
print(f"\n========== 难例 60 条子集对比 ==========")
print(f"Ensemble 准确率:   {(hard_labels == hard_ens_preds).mean():.4f}")
print(f"DeepSeek 准确率:   {(hard_labels[hard_llm_preds != -1] == hard_llm_preds[hard_llm_preds != -1]).mean():.4f}")
print(f"Cascade 准确率:    {(hard_labels == cascade_preds[[s['idx'] for s in hard_samples]]).mean():.4f}")

# ========== 6. 成本与延迟估算 ==========
# DeepSeek API 定价
price_input = 0.5 / 1_000_000   # ¥/token
price_output = 2.0 / 1_000_000

llm_cost = llm_input_tokens * price_input + llm_output_tokens * price_output
llm_cost_per_1200 = llm_cost * (1200 / len(hard_samples))  # 按 5% 路由率折算
print(f"\n========== 成本与延迟 ==========")
print(f"难例调用 {len(hard_samples)} 次，消耗 input={llm_input_tokens}, output={llm_output_tokens} tokens")
print(f"实际成本: ¥{llm_cost:.4f}")
print(f"折算到 1200 条: ¥{llm_cost_per_1200:.4f}")
print(f"LLM 平均延迟: {np.mean(llm_latencies):.3f}s | P95: {np.percentile(llm_latencies, 95):.3f}s")

# 端到端平均延迟（5% 走 LLM，95% 走 BERT）
bert_p95 = 0.00988   # 秒
bert_avg = 0.00876   # 秒
llm_avg = np.mean(llm_latencies)
end_to_end_avg = 0.95 * bert_avg + 0.05 * llm_avg
end_to_end_p95 = 0.95 * bert_p95 + 0.05 * np.percentile(llm_latencies, 95)
print(f"端到端平均延迟: {end_to_end_avg*1000:.1f} ms")
print(f"端到端 P95 延迟: {end_to_end_p95*1000:.1f} ms")

# ========== 7. 保存 cascade_metrics.json ==========
final = {
    "baseline": baseline_metrics,
    "cascade": cascade_metrics,
    "hard_subset": {
        "n_hard": len(hard_samples),
        "ensemble_acc": round(float((hard_labels == hard_ens_preds).mean()), 4),
        "llm_acc": round(float((hard_labels[hard_llm_preds != -1] == hard_llm_preds[hard_llm_preds != -1]).mean()), 4),
        "cascade_acc": round(float((hard_labels == cascade_preds[[s['idx'] for s in hard_samples]]).mean()), 4)
    },
    "cost": {
        "hard_call_cost_yuan": round(float(llm_cost), 4),
        "normalized_to_1200_yuan": round(float(llm_cost_per_1200), 4)
    },
    "latency": {
        "llm_avg_s": round(float(llm_avg), 4),
        "llm_p95_s": round(float(np.percentile(llm_latencies, 95)), 4),
        "end_to_end_avg_ms": round(float(end_to_end_avg * 1000), 2),
        "end_to_end_p95_ms": round(float(end_to_end_p95 * 1000), 2)
    },
    "parse_fail": fail_count
}

with open(f"{BASE}/results/cascade_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到: results/cascade_metrics.json")