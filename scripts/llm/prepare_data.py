import json
from datasets import load_dataset

print("正在准备标准测试集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")
test_data = dataset["test"]

with open("data/test.jsonl", "w", encoding="utf-8") as f:
    for i, item in enumerate(test_data):
        record = {
            "id": i,
            "text": item["text"],
            "label": item["label"]  # 0负面 1正面
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"已生成 test.jsonl，共 {len(test_data)} 条数据。")