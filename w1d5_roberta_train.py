import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

dataset = load_dataset("lansinuote/ChnSentiCorp")
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset = tokenized_datasets["train"].select(range(9600))
eval_dataset = tokenized_datasets["validation"].select(range(1200))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, collate_fn=data_collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


save_path = "./best_saved_model_roberta"
best_f1 = 0.0
patience = 2 
patience_counter = 0
epochs = 5

fgm = FGM(model)

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1} ---")
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        
        fgm.attack() 
        outputs_adv = model(**batch) 
        loss_adv = outputs_adv.loss
        loss_adv.backward() 
        fgm.restore() 
        
        optimizer.step()
        total_loss += loss.item()
    
    print(f"训练集平均 Loss: {total_loss / len(train_dataloader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
    

    current_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"验证集 Macro-F1: {current_f1:.4f}")
    

    if current_f1 > best_f1:
        best_f1 = current_f1
        patience_counter = 0
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"F1提升,模型已保存到 {save_path}")
    else:
        patience_counter += 1
        print(f"F1未提升，忍耐计数: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("触发早停，停止训练")
            break


best_model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)
best_model.eval()

final_preds, final_labels = [], []
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = best_model(**batch)
    preds = torch.argmax(outputs.logits, dim=-1)
    final_preds.extend(preds.cpu().numpy())
    final_labels.extend(batch["labels"].cpu().numpy())


print("\n--- 分类报告 (Classification Report) ---")
print(classification_report(final_labels, final_preds, target_names=["负面(0)", "正面(1)"]))


print("\n--- 混淆矩阵 (Confusion Matrix) ---")
print(confusion_matrix(final_labels, final_preds))

