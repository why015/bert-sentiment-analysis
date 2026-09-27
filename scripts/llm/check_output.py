import json

path = "results/predictions_deepseek-flash_zero_shot.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"真实标签: {data['true_label']}")
        print(f"模型原始输出: [{data['raw_output']}]")
        print(f"解析结果: {data['pred_label']}")
        print("-" * 40)