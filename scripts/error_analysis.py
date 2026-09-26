import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_bert = AutoModelForSequenceClassification.from_pretrained("./best_saved_model").to(device)
model_roberta = AutoModelForSequenceClassification.from_pretrained("./best_saved_model_roberta").to(device)
model_bert.eval()
model_roberta.eval()

dataset = load_dataset("lansinuote/ChnSentiCorp")
test_texts = dataset["test"]["text"] 
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = tokenized_datasets["test"]
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)

all_labels = []
ensemble_preds = []

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["labels"].cpu().numpy()
    all_labels.extend(labels)
    with torch.no_grad():
        probs_bert = torch.softmax(model_bert(**batch).logits, dim=-1).cpu().numpy()
        probs_roberta = torch.softmax(model_roberta(**batch).logits, dim=-1).cpu().numpy()

    ensemble_probs = (probs_bert + probs_roberta) / 2.0
    ensemble_preds.extend(np.argmax(ensemble_probs, axis=-1))


all_labels = np.array(all_labels)
ensemble_preds = np.array(ensemble_preds)

fp_indices = np.where((all_labels == 1) & (ensemble_preds == 0))[0] 
fn_indices = np.where((all_labels == 0) & (ensemble_preds == 1))[0]

print("=== 误报案例 (FP) - 真实: 正面, 预测: 负面 ===")
for i in fp_indices[:]:  
    print(f"文本: {test_texts[i]}")

print("\n=== 漏报案例 (FN) - 真实: 负面, 预测: 正面 ===")
for i in fn_indices[:]: 
    print(f"文本: {test_texts[i]}")