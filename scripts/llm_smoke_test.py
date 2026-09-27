from openai import OpenAI

client = OpenAI(
    api_key="46f9a2cbdd344b90b619287c9767447e.WoqNu4ddhijZ0bMc",  
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

response = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "你是一个情感分析助手。"},
        {"role": "user", "content": "请判断这句话的情感是正面还是负面，只输出'正面'或'负面'。\n\n评论：这家酒店环境太差了，绝对不会再来！"}
    ],
    temperature=0
)

print("模型回复：", response.choices[0].message.content)