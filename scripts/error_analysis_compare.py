import json
import numpy as np

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"

# ========== 1. 加载 Ensemble 预测 ==========
ens_preds = np.load(f"{BASE}/results/ensemble_preds.npy")
labels = np.load(f"{BASE}/results/ensemble_labels.npy")

# ========== 2. 加载 DeepSeek 预测 ==========
llm_preds = []
llm_texts = []
with open(f"{BASE}/results/predictions_deepseek-flash_zero_shot.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        llm_preds.append(d["pred_label"])
        llm_texts.append(d["text"])
llm_preds = np.array(llm_preds)

# ========== 3. 确认样本数一致 ==========
print(f"Ensemble 样本数: {len(ens_preds)}")
print(f"DeepSeek 样本数: {len(llm_preds)}")
print(f"标签样本数: {len(labels)}")
assert len(ens_preds) == len(llm_preds) == len(labels), "样本数不一致！"

# ========== 4. 找出差异样本 ==========
bert_wrong_llm_right = []   # BERT错、LLM对
llm_wrong_bert_right = []   # LLM错、BERT对

for i in range(len(labels)):
    bert_ok = (ens_preds[i] == labels[i])
    llm_ok = (llm_preds[i] == labels[i])
    if not bert_ok and llm_ok:
        bert_wrong_llm_right.append({
            "idx": i,
            "text": llm_texts[i],
            "true_label": int(labels[i]),
            "bert_pred": int(ens_preds[i]),
            "llm_pred": int(llm_preds[i])
        })
    elif bert_ok and not llm_ok:
        llm_wrong_bert_right.append({
            "idx": i,
            "text": llm_texts[i],
            "true_label": int(labels[i]),
            "bert_pred": int(ens_preds[i]),
            "llm_pred": int(llm_preds[i])
        })

# ========== 5. 输出统计 ==========
print(f"\n========== 差异统计 ==========")
print(f"BERT 错、LLM 对: {len(bert_wrong_llm_right)} 条")
print(f"LLM 错、BERT 对: {len(llm_wrong_bert_right)} 条")
print(f"两者都错: {sum(1 for i in range(len(labels)) if ens_preds[i] != labels[i] and llm_preds[i] != labels[i])} 条")
print(f"两者都对: {sum(1 for i in range(len(labels)) if ens_preds[i] == labels[i] and llm_preds[i] == labels[i])} 条")

# ========== 6. 保存结果 ==========
output = {
    "summary": {
        "bert_wrong_llm_right_count": len(bert_wrong_llm_right),
        "llm_wrong_bert_right_count": len(llm_wrong_bert_right)
    },
    "bert_wrong_llm_right": bert_wrong_llm_right,
    "llm_wrong_bert_right": llm_wrong_bert_right
}
with open(f"{BASE}/results/error_analysis_compare.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ========== 7. 打印前 10 条供人工阅读 ==========
print("\n========== BERT 错、LLM 对（前 10 条） ==========")
for s in bert_wrong_llm_right[:10]:
    true_str = "正面" if s["true_label"] == 1 else "负面"
    print(f"[{s['idx']}] 真实={true_str} | BERT={s['bert_pred']} | LLM={s['llm_pred']}")
    print(f"  {s['text'][:120]}")
    print()

print("\n========== LLM 错、BERT 对（前 10 条） ==========")
for s in llm_wrong_bert_right[:10]:
    true_str = "正面" if s["true_label"] == 1 else "负面"
    print(f"[{s['idx']}] 真实={true_str} | BERT={s['bert_pred']} | LLM={s['llm_pred']}")
    print(f"  {s['text'][:120]}")
    print()

print(f"\n完整结果已保存到: results/error_analysis_compare.json")