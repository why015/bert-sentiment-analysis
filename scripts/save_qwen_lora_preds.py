import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import json

BASE = r"C:\Users\123\Desktop\Anaconda Projects\ai_learning\bert-sentiment-analysis"
UNIFIED = f"{BASE}/results/unified"
MODEL_PATH = f"{BASE}/checkpoints/qwen_merged_sentiment"

os.makedirs(UNIFIED, exist_ok=True)

# ========== 1. 加载模型（4-bit 量化） ==========
print("正在加载 Qwen2.5-3B + LoRA 合并模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto"
)
model.eval()

# ========== 2. 加载测试集 ==========
samples = []
with open(f"{BASE}/data/test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"测试集样本数: {len(samples)}")

# ========== 3. Alpaca 格式 Prompt（与训练一致） ==========
PROMPT_TEMPLATE = """### 指令:
请判断以下评论的情感是正面还是负面。只输出'正面'或'负面'。

### 输入:
{text}

### 输出:
"""

# ========== 4. 推理 ==========
preds, labels = [], []
latencies = []
fail = 0

print("\n开始推理...")
for sample in tqdm(samples):
    prompt = PROMPT_TEMPLATE.format(text=sample["text"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    latencies.append(time.time() - start)
    
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    if "正面" in generated and "负面" not in generated:
        pred = 1
    elif "负面" in generated and "正面" not in generated:
        pred = 0
    else:
        pos_idx = generated.rfind("正面")
        neg_idx = generated.rfind("负面")
        if pos_idx > neg_idx and pos_idx != -1:
            pred = 1
        elif neg_idx > pos_idx and neg_idx != -1:
            pred = 0
        else:
            pred = -1
            fail += 1
    
    preds.append(pred)
    labels.append(sample["label"])

# ========== 5. 保存 ==========
preds = np.array(preds)
labels = np.array(labels)

np.save(f"{UNIFIED}/qwen_lora_preds.npy", preds)
np.save(f"{UNIFIED}/qwen_lora_labels.npy", labels)

# ========== 6. 汇总 ==========
valid = preds != -1
acc = (preds[valid] == labels[valid]).mean()
print(f"\n========== 结果 ==========")
print(f"有效样本: {valid.sum()} / {len(preds)}")
print(f"解析失败: {fail}")
print(f"准确率: {acc:.4f}")
print(f"平均延迟: {np.mean(latencies):.3f}s | P95: {np.percentile(latencies, 95):.3f}s")
print(f"\n已保存到: {UNIFIED}")