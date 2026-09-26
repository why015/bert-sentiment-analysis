# FGM 本质上是对抗训练（Adversarial Training）的一种。
# 它在 Embedding 层引入最坏情况下的梯度扰动，迫使模型在输入发生微小变化时也能保持预测稳定。
# 这在数学上等价于一种隐式的数据增强和正则化。
# 它平滑了损失曲面的局部极小值，防止模型过拟合特定的词汇特征，从而提升了模型在验证集上的泛化能力，特别是对难分类别的 F1 指标有明显提升。


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 遍历模型参数，找到Embedding层
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 1. 存档：保存原始权重
                self.backup[name] = param.data.clone()
                # 2. 计算梯度范数
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 3. 核心公式：沿着梯度方向添加扰动
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 恢复原始的Embedding权重
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ====== 实例化 FGM ======
fgm = FGM(model)

# ====== 训练循环内的调用 ======
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 1. 正常前向传播与反向传播
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        
        # ====== 2. FGM 对抗训练核心逻辑 ======
        fgm.attack() # 弄脏 Embedding
        outputs_adv = model(**batch) # 用脏 Embedding 重新前向传播
        loss_adv = outputs_adv.loss
        loss_adv.backward() # 累加对抗梯度（注意不能 zero_grad）
        fgm.restore() # 擦干净 Embedding
        # ===================================
        
        # 3. 更新参数（此时包含正常梯度 + 对抗梯度）
        optimizer.step()
        total_loss += loss.item()