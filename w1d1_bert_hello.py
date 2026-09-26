import torch
from transformers import AutoTokenizer,AutoModel

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "你好,我叫小明"

inputs = tokenizer(text,return_tensors = "pt",padding = True,truncation =True)

print("input_ids:",inputs["input_ids"])
print("attention_mask:",inputs["attention_mask"])
print("token_type_ids:",inputs["token_type_ids"])

model.eval()
with torch.no_grad():
    output = model(**inputs)

