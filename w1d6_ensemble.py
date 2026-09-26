import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

# 1. 加载分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载两个模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在加载模型...")
model_bert = AutoModelForSequenceClassification.from_pretrained("./best_saved_model").to(device)
model_roberta = AutoModelForSequenceClassification.from_pretrained("./best_saved_model_roberta").to(device)
model_bert.eval()
model_roberta.eval()

# 3. 准备测试集数据 (使用 test 数据集)
dataset = load_dataset("lansinuote/ChnSentiCorp")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = tokenized_datasets["test"]
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)

# 4. 收集两个模型的预测概率
all_labels = []
bert_probs = []
roberta_probs = []

print("开始推理...")
for batch in tqdm(test_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["labels"].cpu().numpy()
    all_labels.extend(labels)
    
    with torch.no_grad():
        # BERT 预测
        logits_bert = model_bert(**batch).logits
        probs_bert = torch.softmax(logits_bert, dim=-1).cpu().numpy()
        bert_probs.extend(probs_bert)
        
        # RoBERTa 预测
        logits_roberta = model_roberta(**batch).logits
        probs_roberta = torch.softmax(logits_roberta, dim=-1).cpu().numpy()
        roberta_probs.extend(probs_roberta)

bert_probs = np.array(bert_probs)
roberta_probs = np.array(roberta_probs)

# 5. 评估单模型表现
bert_preds = np.argmax(bert_probs, axis=-1)
roberta_preds = np.argmax(roberta_probs, axis=-1)
print(f"\n测试集 BERT+FGM Macro-F1: {f1_score(all_labels, bert_preds, average='macro'):.4f}")
print(f"测试集 RoBERTa Macro-F1: {f1_score(all_labels, roberta_preds, average='macro'):.4f}")

# 6. 软投票融合 (按 0.5:0.5 权重平均概率)
ensemble_probs = (bert_probs + roberta_probs) / 2.0
ensemble_preds = np.argmax(ensemble_probs, axis=-1)

# 7. 打印最终融合报告
print("\n--- 融合后 (Ensemble) 分类报告 ---")
print(classification_report(all_labels, ensemble_preds, target_names=["负面(0)", "正面(1)"]))

print("\n--- 融合后混淆矩阵 ---")
print(confusion_matrix(all_labels, ensemble_preds))
print(f"\n融合后最终测试集 Macro-F1: {f1_score(all_labels, ensemble_preds, average='macro'):.4f}")