# -*- coding: utf-8 -*-
import json
import os
import torch
import glob
from tqdm import tqdm
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

# --- 配置区 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "valid")
VECTOR_CACHE_DIR = os.path.join(BASE_DIR, "..", "output", "vector_cache")
WHOLE_AUTHOR_PATH = os.path.join(DATA_DIR, "whole_author_profiles.json")
WHOLE_PUB_PATH = os.path.join(DATA_DIR, "whole_author_profiles_pub.json")
TEST_AUTHOR_NUM = None # 仅处理前100个作者用于测试，正式运行时设为 None 或较大值
os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

# --- 模型加载 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# 自动寻找本地路径逻辑保持不变
snapshot_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*")
snapshot_paths = glob.glob(snapshot_pattern)
snapshot_path = snapshot_paths[0] if snapshot_paths else "BAAI/bge-m3"

print(f"正在加载模型: {snapshot_path}")
MODEL = SentenceTransformer(snapshot_path, device=device)
MODEL.max_seq_length = 256
MODEL.half() # 针对3060显存优化

def run_preprocessing():
    # 1. 加载基础数据库
    print("加载数据库中...")
    with open(WHOLE_AUTHOR_PATH, 'r', encoding='utf-8') as f:
        author_db = json.load(f)
    with open(WHOLE_PUB_PATH, 'r', encoding='utf-8') as f:
        whole_pub_db = json.load(f)

    # 2. 遍历所有作者
    all_auth_ids = list(author_db.keys())
    if TEST_AUTHOR_NUM:
        all_auth_ids = all_auth_ids[:TEST_AUTHOR_NUM]
    print(f"共发现 {len(all_auth_ids)} 个作者，开始离线计算向量...")

    for auth_id in tqdm(all_auth_ids):
        cache_path = os.path.join(VECTOR_CACHE_DIR, f"{auth_id}.safetensors")
        
        # 如果已经存在则跳过，方便断点续传
        if os.path.exists(cache_path):
            continue

        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        pub_texts = []
        for pid in pub_ids:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            # 文本构建逻辑必须与你主程序一致！
            text = f"{pub_detail.get('title','')} {' '.join(pub_detail.get('keywords',[]))}".strip()
            if text:
                pub_texts.append(text)

        if not pub_texts:
            continue

        # 3. 批量推理
        try:
            cand_embeddings = MODEL.encode(
                pub_texts,
                batch_size=32, # 离线模式可以调大 batch_size 提高吞吐量
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).half()

            # 4. 使用 safetensors 保存
            save_file({"embeddings": cand_embeddings}, cache_path)
            
        except Exception as e:
            print(f"处理作者 {auth_id} 时出错: {e}")

if __name__ == "__main__":
    run_preprocessing()