import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== 配置 ==========
MODEL_PATH = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis\checkpoints\bert_fgm_seed42"
WARMUP = 10      # 预热次数（排除首次编译开销）
RUNS = 100       # 正式计时次数
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

text = "这家酒店环境太差了，绝对不会再来！"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

# 预热
print("预热中...")
for _ in range(WARMUP):
    with torch.no_grad():
        model(**inputs)

# 正式计时
print(f"开始计时，共 {RUNS} 次...")
latencies = []
for _ in range(RUNS):
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize() if device.type == "cuda" else None
    latencies.append((time.perf_counter() - start) * 1000)  # 转毫秒

latencies = np.array(latencies)
print("\n========== 延迟测试结果 ==========")
print(f"平均延迟 (P50): {np.percentile(latencies, 50):.2f} ms")
print(f"P90 延迟:       {np.percentile(latencies, 90):.2f} ms")
print(f"P95 延迟:       {np.percentile(latencies, 95):.2f} ms")
print(f"P99 延迟:       {np.percentile(latencies, 99):.2f} ms")
print(f"最小/最大:      {latencies.min():.2f} / {latencies.max():.2f} ms")