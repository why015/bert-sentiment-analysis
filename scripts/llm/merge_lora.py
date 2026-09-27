import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 配置 ==========
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "checkpoints/qwen_lora_sentiment"
MERGED_PATH = "checkpoints/qwen_merged_sentiment"
# ==========================

print("正在加载基础模型（CPU, float16）...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",  # 强制用CPU合并，避免爆显存
)

print("正在加载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("正在合并权重...")
merged_model = model.merge_and_unload()

print("正在保存完整模型...")
merged_model.save_pretrained(MERGED_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_PATH)

print(f"✅ 合并完成！完整模型已保存到 {MERGED_PATH}")