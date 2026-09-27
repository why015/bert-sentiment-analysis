import json
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# 改成你的文件路径
path = "results/predictions_deepseek-flash_few_shot.jsonl"

y_true, y_pred = [], []
fail = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data["pred_label"] == -1:
            fail += 1
            continue
        y_true.append(data["true_label"])
        y_pred.append(data["pred_label"])

print(f"有效样本: {len(y_true)} | 解析失败: {fail}")
print(classification_report(y_true, y_pred, target_names=["负面(0)", "正面(1)"]))
print("混淆矩阵:")
print(confusion_matrix(y_true, y_pred))
print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"负面 F1: {f1_score(y_true, y_pred, pos_label=0, average='binary'):.4f}")
print(f"正面 F1: {f1_score(y_true, y_pred, pos_label=1, average='binary'):.4f}")