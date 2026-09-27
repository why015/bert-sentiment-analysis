import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
UNIFIED = f"{BASE}/results/unified"
os.makedirs(UNIFIED, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试集
dataset = load_dataset("lansinuote/ChnSentiCorp")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_labels = np.array(dataset["test"]["label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_loader = DataLoader(tokenized["test"], batch_size=32, collate_fn=data_collator)

np.save(f"{UNIFIED}/test_labels.npy", test_labels)
print(f"✅ test_labels.npy 已保存: shape={test_labels.shape}\n")

# ========== 处理两个单模型 ==========
models_to_process = [
    ("bert_fgm",     f"{BASE}/checkpoints/bert_fgm_seed42"),
    ("roberta_fgm",  f"{BASE}/checkpoints/roberta_seed42"),
]

for name, model_path in models_to_process:
    print(f"正在推理: {name}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        
        all_probs, all_preds = [], []
        for batch in tqdm(test_loader, desc=name):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        np.save(f"{UNIFIED}/{name}_probs.npy", all_probs)
        np.save(f"{UNIFIED}/{name}_preds.npy", all_preds)
        np.save(f"{UNIFIED}/{name}_labels.npy", test_labels)
        
        acc = (all_preds == test_labels).mean()
        print(f"  ✅ {name}: 准确率={acc:.4f}, 保存完成\n")
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ❌ {name} 失败: {e}\n")

print("=" * 60)
print(f"结果已保存到: {UNIFIED}")
print("=" * 60)
for f in sorted(os.listdir(UNIFIED)):
    print(f"  {f}")