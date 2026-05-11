import requests

# 替换为你 config.py 里的 api_key 和 api_base
api_key = "sk-2cjzsGBkq75MUEXM3rhuUA"
api_base = "https://models.sjtu.edu.cn/api/v1" # 例如 https://api.deepseek.com/v1 或其他转发地址

headers = {"Authorization": f"Bearer {api_key}"}
try:
    # 注意：请求的是 /models 路径
    response = requests.get(f"{api_base.rstrip('/')}/models", headers=headers)
    if response.status_code == 200:
        models = response.json()
        print("你当前可用的模型列表：")
        for model in models.get('data', []):
            print(f"- {model['id']}")
    else:
        print(f"请求失败，状态码：{response.status_code}, 错误信息：{response.text}")
except Exception as e:
    print(f"发生异常：{e}")