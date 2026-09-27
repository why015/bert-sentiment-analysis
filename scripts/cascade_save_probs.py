import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. 加载两个模型 ==========
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model_bert = AutoModelForSequenceClassification.from_pretrained(f"{BASE}/checkpoints/bert_fgm_seed42").to(device)
model_roberta = AutoModelForSequenceClassification.from_pretrained(f"{BASE}/checkpoints/roberta_seed42").to(device)
model_bert.eval()
model_roberta.eval()

# ========== 2. 加载测试集 ==========
dataset = load_dataset("lansinuote/ChnSentiCorp")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_loader = DataLoader(tokenized["test"], batch_size=32, collate_fn=data_collator)

# ========== 3. 推理，收集每条样本的概率 ==========
all_probs, all_preds = [], []

for batch in tqdm(test_loader, desc="Ensemble 推理"):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        probs_bert = torch.softmax(model_bert(**batch).logits, dim=-1)
        probs_roberta = torch.softmax(model_roberta(**batch).logits, dim=-1)
        probs = (probs_bert + probs_roberta) / 2.0  # 软投票融合
    all_probs.extend(probs.cpu().numpy())
    all_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(test_labels)

# ========== 4. 计算置信度 = 最大概率 ==========
confidence = np.max(all_probs, axis=-1)

# ========== 5. 保存所有中间结果 ==========
np.save(f"{BASE}/results/ensemble_probs.npy", all_probs)
np.save(f"{BASE}/results/ensemble_preds.npy", all_preds)
np.save(f"{BASE}/results/ensemble_labels.npy", all_labels)
np.save(f"{BASE}/results/ensemble_confidence.npy", confidence)

# ========== 6. 统计置信度分布 ==========
print("\n========== 置信度分布 ==========")
print(f"总样本数: {len(confidence)}")
print(f"置信度: min={confidence.min():.4f}, mean={confidence.mean():.4f}, max={confidence.max():.4f}")
print(f"置信度 < 0.9: {(confidence < 0.9).sum()} 条")
print(f"置信度 < 0.8: {(confidence < 0.8).sum()} 条")
print(f"置信度 < 0.7: {(confidence < 0.7).sum()} 条")

# ========== 7. 找出 5% 最难样本 ==========
threshold = np.percentile(confidence, 5)
hard_indices = np.where(confidence <= threshold)[0]
print(f"\n5% 阈值: {threshold:.4f}")
print(f"难例数: {len(hard_indices)}")

# 保存难例供 LLM 路由使用
hard_samples = []
for idx in hard_indices:
    hard_samples.append({
        "idx": int(idx),
        "text": test_texts[idx],
        "true_label": int(all_labels[idx]),
        "ensemble_pred": int(all_preds[idx]),
        "confidence": float(confidence[idx])
    })

with open(f"{BASE}/results/hard_samples_5pct.json", "w", encoding="utf-8") as f:
    json.dump(hard_samples, f, ensure_ascii=False, indent=2)

# ========== 8. 统计难例中的错误率 ==========
hard_labels = all_labels[hard_indices]
hard_preds = all_preds[hard_indices]
hard_acc = (hard_labels == hard_preds).mean()
print(f"\n难例中 Ensemble 的准确率: {hard_acc:.4f}")
print(f"难例中被 Ensemble 判错的样本数: {(hard_labels != hard_preds).sum()} / {len(hard_indices)}")

# 整体准确率作为对照
overall_acc = (all_labels == all_preds).mean()
print(f"整体准确率: {overall_acc:.4f}")
print(f"\n结果已保存到 {BASE}/results/")