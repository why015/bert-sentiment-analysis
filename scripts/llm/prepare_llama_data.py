import json
from datasets import load_dataset

dataset = load_dataset("lansinuote/ChnSentiCorp")
train_data = dataset["train"]

data_list = []
for item in train_data:
    label = "正面" if item["label"] == 1 else "负面"
    data_list.append({
        "instruction": "请判断以下评论的情感是正面还是负面。只输出'正面'或'负面'。",
        "input": item["text"],
        "output": label
    })

with open("data/llm_train_data.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"已生成 data/llm_train_data.json，共 {len(data_list)} 条数据。")