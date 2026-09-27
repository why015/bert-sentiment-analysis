import os
import time
import numpy as np
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

API_KEY = os.getenv("DEEPSEEK_API_KEY", "") 
MODEL_NAME = "deepseek-flash"
SAMPLE_SIZE = 100         
SLEEP_INTERVAL = 0       
MAX_RETRIES = 5           

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

dataset = load_dataset("lansinuote/ChnSentiCorp")
test_data = dataset["test"].select(range(SAMPLE_SIZE))
texts = test_data["text"]
labels = test_data["label"]

SYSTEM_PROMPT = "你是一个专业的中文情感分析助手。"
USER_TEMPLATE = """请判断以下评论的情感是正面还是负面。只输出"正面"或"负面"，不要输出任何其他内容。

评论：{text}"""

def predict_sentiment(text):
    """调用LLM进行零样本情感预测，带自动重试"""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(text=text)}
                ],
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            if "正面" in result:
                return 1
            elif "负面" in result:
                return 0
            else:
                return -1
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "1302" in err_msg or "1305" in err_msg:
                # 限流，指数退避等待
                wait_time = SLEEP_INTERVAL * (attempt + 1)
                time.sleep(wait_time)
                continue
            else:
                print(f"API调用出错: {e}")
                return -1
    return -1 

print(f"开始 Zero-shot 评测，共 {SAMPLE_SIZE} 条，请求间隔 {SLEEP_INTERVAL}s ...")
preds = []
latencies = []

for text in tqdm(texts):
    start = time.time()
    pred = predict_sentiment(text)
    latency = time.time() - start
    latencies.append(latency)
    preds.append(pred)
    time.sleep(SLEEP_INTERVAL) 

valid_idx = [i for i, p in enumerate(preds) if p != -1]
valid_preds = [preds[i] for i in valid_idx]
valid_labels = [labels[i] for i in valid_idx]

print(f"\n有效预测数: {len(valid_idx)} / {SAMPLE_SIZE}")
if latencies:
    print(f"平均延迟: {np.mean(latencies):.2f}s")
    print(f"P95延迟: {np.percentile(latencies, 95):.2f}s")
    print(f"总耗时: {sum(latencies)/60:.1f} 分钟")

if len(valid_idx) > 0:
    print("\n--- LLM Zero-shot 分类报告 ---")
    print(classification_report(valid_labels, valid_preds, target_names=["负面(0)", "正面(1)"]))
    print("混淆矩阵:")
    print(confusion_matrix(valid_labels, valid_preds))
    print(f"\nLLM Zero-shot Macro-F1: {f1_score(valid_labels, valid_preds, average='macro'):.4f}")