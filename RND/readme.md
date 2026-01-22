main.py
从whole论文库里拿到论文信息 → 定位这篇论文里第 index 个作者 → 在作者简历库里选出最可能的作者ID → 把答案写入 output/result.json
cna_valid_unass.json：待消歧题目列表（形如 paperID-index）

cna_valid_unass_pub.json：论文库（paperID → 论文详细信息）

whole_author_profiles.json：作者简历库（authorID → 作者profile）

output/result.json：你要提交/评测的预测输出

# 1. 创建新环境 (建议 Python 3.9 或以上)
conda create -n rnd_dspy python=3.10

# 2. 激活环境
conda activate rnd_dspy

# 3. 安装核心依赖
pip install dspy-ai openai tqdm  transformers
