# 基于 BERT 的中文情感分析系统

## 项目简介

本项目基于 `bert-base-chinese` 预训练模型，针对中文酒店评论数据集 `ChnSentiCorp` 进行情感分析微调。

项目涵盖数据处理、模型训练、深度评估和 FastAPI 工程部署，并引入早停机制与 FGM 对抗训练。

## 核心工作

- **数据工程**：使用 Hugging Face `datasets` 库进行批量分词，采用 `DataCollatorWithPadding` 实现动态填充。
- **模型微调**：手写 PyTorch 训练循环，使用 `AdamW` 优化器，学习率设为 `2e-5`，结合早停机制防止过拟合。
- **对抗训练**：实现 FGM 对抗训练，在 Embedding 层添加梯度扰动，累加正常梯度与对抗梯度。
- **深度评估**：采用 Precision、Recall、Macro-F1 及混淆矩阵评估模型表现。
- **工程部署**：基于 FastAPI 封装推理接口，利用 Pydantic 进行数据验证，自动生成 Swagger 文档。

## 模型效果

| 模型版本 | Macro-F1 | 负面 F1 | 正面 F1 | 说明 |
| --- | --- | --- | --- | --- |
| Baseline（BERT） | 0.9450 | 0.94 | 0.95 | 基础微调 |
| BERT + FGM 对抗训练 | **0.9533** | 0.95 | 0.95 | 最优模型 |

> 在本次评估中，FGM 将误报（FP）从 38 降低至 28。

## 项目结构

```text
.
├── w1d1_bert_hello.py     # BERT 基础加载与推理验证
├── w1d2_train.py          # BERT 基础微调训练
├── w1d3_inference.py      # 模型保存与单条文本推理
├── w1d4_deep_train.py     # 核心训练脚本（含早停与 FGM 对抗训练）
├── main.py                # FastAPI 推理服务入口
├── requirements.txt       # 环境依赖
├── .gitignore             # Git 忽略文件配置
└── README.md              # 项目说明文档
```

## 环境配置

### 1. 创建并激活虚拟环境

```bash
conda create -n pytorch_env python=3.9
conda activate pytorch_env
```

### 2. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 运行方式

### 1. 模型训练（含早停与 FGM）

```bash
python -X utf8 -u w1d4_deep_train.py
```

### 2. 启动推理服务

```bash
python -m uvicorn main:app --reload --port 8000
```

### 3. 调用 API 接口

服务启动后，打开浏览器访问 [Swagger 交互式文档](http://127.0.0.1:8000/docs)。

**接口：** `POST /predict`

**请求示例：**

```json
{
  "text": "这家酒店环境太差了，绝对不会再来！"
}
```

**响应示例：**

```json
{
  "text": "这家酒店环境太差了，绝对不会再来！",
  "label": "负面",
  "logits": [3.67, -3.49]
}
```

> 响应中的数值仅为示例，实际输出以模型推理结果为准。