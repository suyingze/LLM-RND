# 1. 创建新环境 (建议 Python 3.9 或以上)
conda create -n rnd_dspy python=3.10

# 2. 激活环境
conda activate rnd_dspy

# 3. 安装核心依赖
pip install dspy-ai openai tqdm  transformers
pip install sentence-transformers numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

