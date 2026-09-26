import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
from tqdm import tqdm
dataset = load_dataset("lansinuote/ChnSentiCorp")

print(dataset)

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"],truncation = True,max_length = 128) 

tokenized_datasets = dataset.map(tokenize_function,batched = True,remove_columns=["text"] )

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"],  
    shuffle=True,                 
    batch_size=16,               
    collate_fn=data_collator      
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    batch_size=32,               
    collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train() 
for epoch in range(3):
    for batch in train_dataloader: 
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()      
        optimizer.step()     

metric = evaluate.load("accuracy") 

model.eval() 

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad(): 
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    metric.add_batch(predictions=predictions, references=batch["labels"])

print("验证集准确率:", metric.compute())

save_path = "./saved_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"模型和分词器已保存到 {save_path}")