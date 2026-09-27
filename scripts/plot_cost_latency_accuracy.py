import os
import matplotlib.pyplot as plt
import numpy as np

# ========== 中文字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 数据（按 P95 延迟升序排列） ==========
models = [
    "BERT+Ensemble\n(软投票融合)",
    "Qwen2.5-3B\n(Zero-shot)",
    "Cascade\n(Ensemble+LLM 5%)",
    "Qwen2.5-3B + QLoRA\n(微调)",
    "DeepSeek-V4.1 Flash\n(Zero-shot)"
]
p95_latency = [20, 90, 92, 190, 1570]              # ms
macro_f1 =    [0.9586, 0.8807, 0.9558, 0.9258, 0.8941]
cost =        [0.001, 0.003, 0.32, 0.005, 0.22]    # ¥/千条

# ========== 气泡大小统一（视觉清爽） ==========
sizes = [200] * len(models)

# ========== 颜色 ==========
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']

# ========== 绘图 ==========
fig, ax = plt.subplots(figsize=(12, 7.5))

for i in range(len(models)):
    ax.scatter(
        p95_latency[i], macro_f1[i],
        s=sizes[i], c=colors[i],
        alpha=0.7, edgecolors='black', linewidth=1.8,
        label=models[i].replace('\n', ' ')
    )
    # 数值标注（含成本和 F1）
    ax.annotate(
        f"F1={macro_f1[i]:.4f}\n{cost[i]}元/千条",
        (p95_latency[i], macro_f1[i]),
        textcoords="offset points",
        xytext=(0, -45 if i != 1 else 45),
        ha='center', fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray")
    )

# ========== 坐标轴与刻度 ==========
ax.set_xscale('log')
ax.set_xlabel('P95 延迟 (ms，对数坐标)', fontsize=13, fontweight='bold')
ax.set_ylabel('Macro-F1', fontsize=13, fontweight='bold')
ax.set_title('成本-延迟-精度三维对比 (所有方案基于同一 1200 条测试集)',
             fontsize=14, fontweight='bold', pad=15)

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0.86, 0.98)
ax.set_xlim(10, 2500)

# ========== 图例 ==========
ax.legend(loc='lower left', fontsize=10, framealpha=0.95,
          title='模型 / 方法', title_fontsize=11)

# ========== 右上角图注 ==========
fig.text(0.99, 0.02,
         '气泡统一大小；每个气泡下方标注 F1 与每千条成本',
         ha='right', fontsize=9, color='gray', style='italic')

plt.tight_layout()

# ========== 保存 ==========
os.makedirs("results", exist_ok=True)
save_path = "results/cost_latency_accuracy.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"图表已保存到: {save_path}")

plt.show()