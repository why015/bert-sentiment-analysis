import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_path = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis\checkpoints\bert_fgm_seed42"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path)
model.to(device)
model.eval() 
print("模型加载完毕！\n")

label_map = {0: "负面评价", 1: "正面评价"}

test_texts = [
    "这家酒店环境太差了，房间还有异味，绝对不会再来！",
    "房间很干净，服务态度特别好，下次还会入住！",
    "一般般吧，没有想象中那么好，但也凑合。"
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()
    
    print(f"输入文本: {text}")
    print(f"模型预测: {label_map[pred_id]}")
    print(f"原始分数 (logits): {logits.cpu().numpy()[0]}")
    print("-" * 50)