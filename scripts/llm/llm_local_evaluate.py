import os
import json
import time
import yaml
import argparse
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ========== 本地 Ollama 配置 ==========
API_KEY = "ollama"                          # 本地不需要真实 Key
MODEL_NAME = "qwen2.5:3b"                   # 本地模型名
BASE_URL = "http://localhost:11434/v1"      # Ollama 默认端口
# ====================================


def load_data(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def parse_output(raw_output):
    """从 LLM 输出中解析出预测标签：1=正面，0=负面，-1=解析失败"""
    text = raw_output.strip()
    if "正面" in text and "负面" not in text:
        return 1
    if "负面" in text and "正面" not in text:
        return 0
    pos_idx = text.rfind("正面")
    neg_idx = text.rfind("负面")
    if pos_idx > neg_idx:
        return 1
    if neg_idx > pos_idx:
        return 0
    return -1


def main(args):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    with open("prompts/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    prompt_template = prompts[args.prompt_type]

    samples = load_data("data/test.jsonl")
    if args.sample_size > 0:
        samples = samples[:args.sample_size]

    print(f"开始评测 | 模型: {MODEL_NAME} | Prompt: {args.prompt_type} | 样本数: {len(samples)}")

    predictions = []
    y_true, y_pred = [], []
    latencies = []
    fail_count = 0

    for sample in tqdm(samples):
        prompt = prompt_template.format(text=sample["text"])
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64  # 本地模型听话，64 足够
            )
            latency = time.time() - start
            latencies.append(latency)

            raw_output = response.choices[0].message.content.strip()
            # 兜底：某些推理模型会把内容放在 reasoning_content
            if not raw_output:
                reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""
                raw_output = reasoning.strip()

            # 本地 Ollama 的 usage 字段可能不完整，做安全兜底
            input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0

            pred_label = parse_output(raw_output)
            if pred_label == -1:
                fail_count += 1
            else:
                y_true.append(sample["label"])
                y_pred.append(pred_label)

            predictions.append({
                "id": sample["id"],
                "text": sample["text"],
                "true_label": sample["label"],
                "raw_output": raw_output,
                "pred_label": pred_label,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        except Exception as e:
            print(f"\n[Error] id={sample['id']}: {e}")
            time.sleep(1)

    os.makedirs("results", exist_ok=True)
    output_path = f"results/predictions_{MODEL_NAME}_{args.prompt_type}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    parse_fail_rate = fail_count / len(samples) if samples else 0
    metrics = {
        "model": MODEL_NAME,
        "prompt_type": args.prompt_type,
        "sample_size": len(samples),
        "parse_fail_rate": round(parse_fail_rate, 4),
        "avg_latency": round(float(np.mean(latencies)), 4) if latencies else 0,
        "p95_latency": round(float(np.percentile(latencies, 95)), 4) if latencies else 0,
        "avg_input_tokens": round(float(np.mean([p["input_tokens"] for p in predictions])), 2) if predictions else 0,
        "avg_output_tokens": round(float(np.mean([p["output_tokens"] for p in predictions])), 2) if predictions else 0,
    }
    if len(y_true) > 0:
        metrics["macro_f1"] = round(float(f1_score(y_true, y_pred, average="macro")), 4)
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)

    metrics_path = f"results/metrics_{MODEL_NAME}_{args.prompt_type}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n========== 评测结果 ==========")
    print(f"解析失败率: {parse_fail_rate * 100:.2f}%")
    if latencies:
        print(f"平均延迟: {np.mean(latencies):.2f}s | P95延迟: {np.percentile(latencies, 95):.2f}s")
    if len(y_true) > 0:
        print(classification_report(y_true, y_pred, target_names=["负面(0)", "正面(1)"]))
        print(f"Macro-F1: {metrics['macro_f1']}")
    print(f"\n结果已保存到: {output_path}")
    print(f"指标已保存到: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot", "json"])
    parser.add_argument("--sample_size", type=int, default=100,
                        help="评测样本量，-1 表示全量 1200 条")
    args = parser.parse_args()
    main(args)