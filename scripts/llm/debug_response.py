import os
from openai import OpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "请判断这句话的情感是正面还是负面，只输出'正面'或'负面'。\n\n评论：这家酒店太好了！"}],
    temperature=0,
    max_tokens=64
)

print("=" * 50)
print("完整响应对象:")
print(response)
print("=" * 50)
print("content 字段:", repr(response.choices[0].message.content))
print("finish_reason:", response.choices[0].finish_reason)
print("=" * 50)
# 打印 message 对象的所有属性
print("message 对象所有字段:")
print(response.choices[0].message.model_dump())