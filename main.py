from fastapi import FastAPI
from pydantic import BaseModel  # 用于定义请求格式
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== 1. 加载模型（只在启动时执行一次） ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()
print("模型加载完毕！")

# ========== 2. 创建 FastAPI 应用 ==========
app = FastAPI(title="情感分析API")

# ========== 3. 定义请求格式 ==========
# 顾客递进来的纸条，必须包含一个 text 字段（字符串）
class TextRequest(BaseModel):
    text: str

# ========== 4. 定义预测接口 ==========
@app.post("/predict")
def predict(request: TextRequest):  # 这里接收的是一个对象，里面有 text 字段
    # 分词
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()
    
    label_map = {0: "负面", 1: "正面"}
    return {
        "text": request.text,
        "label": label_map[pred_id],
        "logits": logits.cpu().numpy()[0].tolist()
    }