# -*- coding: utf-8 -*-
import json
import os
import torch
import glob
from tqdm import tqdm
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

# --- 0. 环境变量设置 (必须在 import util 之前或最顶部) ---
os.environ["CURRENT_DATASET"] = "sa_lzk"

# 引入你的工具函数
from util import get_vector_cache_path, build_feature_text

# --- 1. 路径配置区 (更新为新数据集路径) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "sa_lzk_data") 

WHOLE_AUTHOR_PATH = os.path.join(DATA_DIR, "profiles", "whole_author_profiles.json") 
WHOLE_PUB_PATH = os.path.join(DATA_DIR, "profiles", "whole_author_profiles_pub.json") 

VECTOR_CACHE_DIR = get_vector_cache_path() 

TEST_AUTHOR_NUM = None # 正式运行时设为 None 遍历全部作者

# --- 2. 模型加载 (保持原样) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
snapshot_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*")
snapshot_paths = glob.glob(snapshot_pattern)
snapshot_path = snapshot_paths[0] if snapshot_paths else "BAAI/bge-m3"

print(f"正在加载模型: {snapshot_path}")
MODEL = SentenceTransformer(snapshot_path, device=device)
MODEL.max_seq_length = 512
MODEL.half() # 针对 3060 等显存进行优化

def run_preprocessing():
    # 1. 加载基础数据库
    print(f"正在加载新数据集数据库: {DATA_DIR}")
    if not os.path.exists(WHOLE_AUTHOR_PATH):
        print(f"错误: 找不到文件 {WHOLE_AUTHOR_PATH}")
        return

    with open(WHOLE_AUTHOR_PATH, 'r', encoding='utf-8') as f:
        author_db = json.load(f)
    with open(WHOLE_PUB_PATH, 'r', encoding='utf-8') as f:
        whole_pub_db = json.load(f)

    # 2. 准备遍历
    all_auth_ids = list(author_db.keys())
    if TEST_AUTHOR_NUM:
        all_auth_ids = all_auth_ids[:TEST_AUTHOR_NUM]
    
    print(f"目标文件夹: {VECTOR_CACHE_DIR}")
    print(f"共发现 {len(all_auth_ids)} 个作者，开始离线计算向量...")

    for auth_id in tqdm(all_auth_ids):
        # 结果将保存到 output/vector_cache/sa_lzk/T_K (或对应模式名) 下
        cache_path = os.path.join(VECTOR_CACHE_DIR, f"{auth_id}.safetensors")
        
        # 断点续传逻辑
        if os.path.exists(cache_path):
            continue

        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        pub_texts = []
        for pid in pub_ids:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            
            # 这里会自动根据 util.py 里的 CURRENT_FEATURE_MODE 拼接文本
            text = build_feature_text(pub_detail)
            if text:
                pub_texts.append(text)

        if not pub_texts:
            continue

        # 3. 批量推理
        try:
            cand_embeddings = MODEL.encode(
                pub_texts,
                batch_size=32, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).half()

            # 4. 保存
            save_file({"embeddings": cand_embeddings}, cache_path)
            
        except Exception as e:
            print(f"处理作者 {auth_id} 时出错: {e}")

if __name__ == "__main__":
    run_preprocessing()