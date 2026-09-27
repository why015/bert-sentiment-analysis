import json
import time
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ========== 配置 ==========
MODEL_PATH = "checkpoints/qwen_merged_sentiment"
SAMPLE_SIZE = 1200  # 全量评测
# ==========================

# 4-bit 量化加载，确保 8GB 显存够用
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

# 【修改1】使用与训练时完全一致的 Alpaca 格式 Prompt
prompt_template = """### 指令:
请判断以下评论的情感是正面还是负面。只输出'正面'或'负面'。

### 输入:
{text}

### 输出:
"""

# 加载测试集
samples = []
with open("data/test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))
samples = samples[:SAMPLE_SIZE]

y_true, y_pred = [], []
latencies = []
fail_count = 0

print(f"开始评测 | 模型: {MODEL_PATH} | 样本数: {len(samples)}")
for sample in tqdm(samples):
    prompt = prompt_template.format(text=sample["text"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start = time.time()
    with torch.no_grad():
        # 【修改2】max_new_tokens 从 10 增大到 20
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    latency = time.time() - start
    latencies.append(latency)
    
    # 只取新生成的部分
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    raw_output = generated.strip()
    
    # 【修改3】增强解析逻辑，优先匹配单一出现的情感词，兜底取最后出现的词
    if "正面" in raw_output and "负面" not in raw_output:
        pred = 1
    elif "负面" in raw_output and "正面" not in raw_output:
        pred = 0
    else:
        pos_idx = raw_output.rfind("正面")
        neg_idx = raw_output.rfind("负面")
        if pos_idx > neg_idx and pos_idx != -1:
            pred = 1
        elif neg_idx > pos_idx and neg_idx != -1:
            pred = 0
        else:
            pred = -1
            fail_count += 1
    
    if pred != -1:
        y_true.append(sample["label"])
        y_pred.append(pred)

# 计算指标
print(f"\n解析失败率: {fail_count / len(samples) * 100:.2f}%")
print(f"平均延迟: {np.mean(latencies):.2f}s | P95延迟: {np.percentile(latencies, 95):.2f}s")
print(classification_report(y_true, y_pred, target_names=["负面(0)", "正面(1)"]))
print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

# 保存结果
result = {
    "model": MODEL_PATH,
    "sample_size": len(samples),
    "parse_fail_rate": round(fail_count / len(samples), 4),
    "p95_latency": round(float(np.percentile(latencies, 95)), 4),
    "macro_f1": round(float(f1_score(y_true, y_pred, average='macro')), 4),
    "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
}
with open("results/metrics_qwen_lora_merged.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n结果已保存到 results/metrics_qwen_lora_merged.json")