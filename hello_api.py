from fastapi import FastAPI

# 1. 创建服务员
app = FastAPI()

# 2. 定义一个 GET 路由（顾客走后门进来，看到了说明书）
@app.get("/")
def read_root():
    return {"message": "欢迎来到情感分析API"}

# 3. 定义一个 POST 路由（顾客从前门递纸条）
@app.post("/greet")
def greet(name: str):
    return {"message": f"你好，{name}！"}