import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

model_path = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis\checkpoints\bert_fgm_seed42"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

dataset = load_dataset("lansinuote/ChnSentiCorp")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32, 
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))

all_preds, all_labels = [], []
for batch in tqdm(test_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
    all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    all_labels.extend(batch["labels"].cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["负面(0)", "正面(1)"]))
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
print(f"\n测试集数据: Accuracy={accuracy_score(all_labels, all_preds):.4f}, FP={fp}, FN={fn}, Macro-F1={f1_score(all_labels, all_preds, average='macro'):.4f}")