import json

path = "results/predictions_deepseek-flash_zero_shot.jsonl"

fails = []
total = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        total += 1
        if data["pred_label"] == -1:
            fails.append(data)

print(f"总样本数: {total}")
print(f"解析失败数: {len(fails)}\n")

for i, f in enumerate(fails):
    print(f"--- 失败样本 {i+1} ---")
    print(f"原文本: {f['text']}")
    print(f"真实标签: {f['true_label']} (1正面/0负面)")
    print(f"模型原始输出: [{f['raw_output']}]")
    print(f"输入Token: {f['input_tokens']} | 输出Token: {f['output_tokens']}")
    print("-" * 50)