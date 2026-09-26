# 基于 BERT 的中文情感分析系统

## 项目简介

本项目基于 `bert-base-chinese` 预训练模型，针对中文酒店评论数据集 `ChnSentiCorp` 进行情感分析微调。

项目涵盖数据处理、模型训练、深度评估和 FastAPI 工程部署，并引入早停机制与 FGM 对抗训练。

### 升级版：多模型融合

针对单模型性能瓶颈，引入多模型融合（Ensemble）。分别微调 `bert-base-chinese` 与 `hfl/chinese-roberta-wwm-ext`，采用 Softmax 概率软投票进行融合。

通过模型互补效应，在统一测试集上最终达到 **Macro-F1 0.9586 ± 0.0008**，各指标均达到最优。

## 核心工作

- **数据工程**：使用 Hugging Face `datasets` 库进行批量分词，采用 `DataCollatorWithPadding` 实现动态填充。
- **模型微调**：手写 PyTorch 训练循环，使用 `AdamW` 优化器，学习率设为 `2e-5`，结合早停机制防止过拟合。
- **对抗训练**：实现 FGM 对抗训练，在 Embedding 层添加梯度扰动，累加正常梯度与对抗梯度。
- **多模型融合**：分别微调 BERT 与 RoBERTa-wwm-ext，将模型输出转换为 Softmax 概率，通过软投票得到最终预测。
- **深度评估**：采用 Precision、Recall、Macro-F1 及混淆矩阵评估模型表现，并进行了完整的消融实验与错误分析。
- **工程部署**：基于 FastAPI 封装推理接口，利用 Pydantic 进行数据验证，自动生成 Swagger 文档。

## 消融实验与多次实验 (Ablation Study & Repeated Experiments)

所有模型均在**统一测试集（Test Set, 1200条）**上进行评估。Baseline 采用固定 3 个 Epoch 训练；加入早停与 FGM 后的模型采用 Early Stopping（patience=2）防止过拟合。核心模型均使用 3 个不同的随机种子（42, 2026, 1234）进行独立重复实验，结果取均值 ± 标准差。

| 模型配置 | 种子 | 测试集 Macro-F1 | 测试集 Accuracy | FP | FN | 说明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| BERT (Baseline) | 42 | 0.9442 | 0.9442 | 26 | 41 | 固定3 Epoch |
| BERT (Baseline) | 2026 | 0.9375 | 0.9375 | 39 | 36 | 固定3 Epoch |
| BERT (Baseline) | 1234 | 0.9458 | 0.9458 | 14 | 51 | 固定3 Epoch |
| **BERT (Baseline 均值)** | **3种子** | **0.9425 ± 0.0036** | **0.9425 ± 0.0036** | - | - | **基准对照组** |
| BERT + Early Stopping | 42 | 0.9442 | 0.9442 | 26 | 41 | 早停训练 |
| BERT + Early Stopping | 2026 | 0.9375 | 0.9375 | 39 | 36 | 早停训练 |
| BERT + Early Stopping | 1234 | 0.9442 | 0.9442 | 28 | 39 | 早停训练 |
| **BERT + Early Stopping (均值)** | **3种子** | **0.9420 ± 0.0032** | **0.9420 ± 0.0032** | - | - | **早停消融（无显著提升）** |
| BERT + ES + FGM | 42 | 0.9608 | 0.9608 | 24 | 23 | 早停 + 对抗训练 |
| BERT + ES + FGM | 2026 | 0.9516 | 0.9517 | 33 | 25 | 早停 + 对抗训练 |
| BERT + ES + FGM | 1234 | 0.9558 | 0.9558 | 20 | 33 | 早停 + 对抗训练 |
| **BERT + ES + FGM (均值)** | **3种子** | **0.9561 ± 0.0038** | **0.9561 ± 0.0038** | - | - | **对抗训练核心增益** |
| RoBERTa + ES + FGM | 42 | 0.9550 | 0.9550 | 26 | 28 | 不同架构 |
| RoBERTa + ES + FGM | 2026 | 0.9558 | 0.9558 | 26 | 27 | 不同架构 |
| RoBERTa + ES + FGM | 1234 | 0.9558 | 0.9558 | 25 | 28 | 不同架构 |
| **RoBERTa + ES + FGM (均值)** | **3种子** | **0.9555 ± 0.0004** | **0.9555 ± 0.0004** | - | - | **架构对比** |
| Ensemble (软投票) | 42对42 | 0.9592 | 0.9592 | 22 | 27 | BERT42 + RoBERTa42 |
| Ensemble (软投票) | 2026对2026 | 0.9592 | 0.9592 | 24 | 25 | BERT2026 + RoBERTa2026 |
| Ensemble (软投票) | 1234对1234 | 0.9575 | 0.9575 | 21 | 30 | BERT1234 + RoBERTa1234 |
| **Ensemble (软投票, 最终版本)** | **3对3对应** | **0.9586 ± 0.0008** | **0.9586 ± 0.0008** | - | - | **最优性能：模型融合** |

> **注**：所有模型均在统一测试集（1200 条）上评估。测试集正负样本接近平衡（592/608），因此 Accuracy 与 Macro-F1 数值接近。消融结论如下：
> 
> 1. **Early Stopping**：均值 0.9420 ± 0.0032，与固定 3 Epoch 的 Baseline 0.9425 ± 0.0036 接近，未直接提升上限，主要作用是防止过拟合与训练不稳定。
> 2. **FGM 对抗训练**：在同种子 ES 基础上，三个种子分别提升 +1.66 / +1.41 / +1.16 pp，平均 +1.41 pp，均值 0.9561 ± 0.0038；三个种子方向一致，是主要增益来源，但显著性仍需更多种子或配对检验验证。
> 3. **RoBERTa**：均值 0.9555 ± 0.0004，对随机种子不敏感，稳定性较好。
> 4. **Ensemble**：软投票融合达到 0.9586 ± 0.0008，相比 RoBERTa 仅提升 0.31 pp，边际收益较小；工程上可按延迟/成本在 RoBERTa 与 Ensemble 之间选择。

## 错误分析 (Error Analysis)

为了进一步探究模型瓶颈，我抽取了 Ensemble 模型在测试集中的 51 条错误样本（29条误报 FP，22条漏报 FN）进行人工分析，归纳出以下四大类问题：

| 错误类型 | 数量 | 占比 | 典型案例与说明 |
| :--- | :---: | :---: | :--- |
| **标签噪声** | 17 | 33% | “早餐太失望了...以后不会再来了”被标注为“正面”。公开数据集存在明显的标注错误，模型在此类样本上被“冤枉”。 |
| **混合情感** | 17 | 33% | “房间隔音太差，赠送的早餐非常好吃”。长文本中褒贬共存，模型容易被局部强烈的情感词带偏。 |
| **领域不匹配/中性** | 16 | 31% | “请问怎么编辑？”“系统能不能二选一”。数据集中混入了大量非酒店评论、疑问句或客观陈述，缺乏明显情感倾向。 |
| **反语/讽刺** | 1 | 2% | “实在是太享受了...希望大家夜不能眠”。表面夸奖实则抱怨，是当前 NLP 模型的普遍短板。 |

**未来优化方向：**
1. **数据清洗**：针对标签噪声，引入置信度学习（Confidence Learning）或人工复核部分训练数据。
2. **长文本处理**：针对混合情感，尝试将 `max_length` 扩大至 512，或引入 Longformer 等长文本模型。
3. **路由架构**：针对中性/模糊文本，引入置信度阈值，低于阈值的难例交由 LLM（大模型）兜底，而不是让 BERT 强行二选一。

## 项目结构

```text
bert-sentiment-analysis/
├── app/
│   └── main.py                 # FastAPI 推理服务入口
├── checkpoints/                # 存放所有训练好的模型权重（通过 .gitignore 忽略，不传Git）
├── scripts/                    # 存放所有入口脚本
│   ├── train_baseline.py       # 基础 BERT 训练
│   ├── train_bert_fgm.py       # BERT + FGM 训练
│   ├── train_roberta_fgm.py    # RoBERTa + FGM 训练
│   ├── ensemble_eval.py        # 多模型融合评估
│   ├── error_analysis.py       # 错误分析
│   ├── evaluate.py             # 通用测试集评估
│   └── inference.py            # 推理演示
├── .gitignore
├── README.md
└── requirements.txt
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

### 1. BERT 模型训练（含早停与 FGM）

```bash
python -X utf8 -u scripts/train_bert_fgm.py
```

### 2. RoBERTa-wwm-ext 模型训练

```bash
python -X utf8 -u scripts/train_roberta_fgm.py
```

### 3. 多模型融合

完成两个模型的训练并保存模型后，运行融合脚本：

```bash
python -X utf8 -u scripts/ensemble_eval.py
```

### 4. 启动推理服务

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 5. 调用 API 接口

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

### 6. 单条文本推理演示

```bash
python -X utf8 -u scripts/inference.py
```