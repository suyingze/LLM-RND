import json
import os
import re
import time
import torch
import numpy as np
from collections import Counter
from typing import Dict, List
from rapidfuzz import fuzz
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# 配置客户端
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", 
    api_key="sk-no-key-required"
)

# --- 预处理函数保持不变 ---

NOISE_PATTERNS = [
    r"school of", r"department of", r"dept of", r"dept\.",
    r"college of", r"institute of", r"lab of", r"laboratory of",
    r"faculty of", r"division of", r"center for", r"centre for"
]

CORE_KEYWORDS = ["university", "institute", "academy", "college", "center", "laboratory", "research", "school"]

ABBR_MAP: Dict[str, str] = {
    "mit": "massachusetts institute of technology",
    "cas": "chinese academy of sciences",
    "ucas": "university of chinese academy of sciences",
    "caltech": "california institute of technology",
    "oxford": "university of oxford",
}

def normalize_org(org: str) -> str:
    if not org: return ""
    s = re.sub(r"\s+", " ", org.strip().lower())
    tokens = re.split(r"[ ,;]+", s)
    mapped = [ABBR_MAP.get(t, t) for t in tokens]
    s = " ".join(mapped)
    for p in NOISE_PATTERNS:
        s = re.sub(p, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    
    core_candidate = ""
    for part in parts:
        if any(kw in part for kw in CORE_KEYWORDS):
            core_candidate = part
            break
    if not core_candidate and parts:
        core_candidate = parts[-1]
    
    core_candidate = re.sub(r"[^a-z0-9 ]", "", core_candidate).strip()
    return core_candidate.title()

def same_name(a: str, b: str) -> bool:
    def _norm(n):
        return re.sub(r"[\.\,\-\s]+", "_", n.strip().lower())
    na, nb = _norm(a), _norm(b)
    if na == nb: return True
    pa, pb = na.split("_"), nb.split("_")
    if len(pa) == 2 and len(pb) == 2:
        if pa[0] == pb[1] and pa[1] == pb[0]: return True
        if (len(pa[0]) == 1 and pb[0].startswith(pa[0]) and pa[1] == pb[1]) or \
           (len(pb[0]) == 1 and pa[0].startswith(pb[0]) and pa[1] == pb[1]): return True
    return False

# --- 核心性能优化部分 ---

def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """
    将整个列表一次性传给服务器，利用 Llama/Qwen 的 Batch 推理能力
    """
    if not texts:
        return np.array([])
    
    # 过滤空值并记录索引，确保返回维度对齐
    cleaned_texts = [t.strip() if t.strip() else "n/a" for t in texts]
    
    try:
        response = client.embeddings.create(
            model="qwen3",
            input=cleaned_texts  # 关键：传入 List 而非 String
        )
        # 提取 embedding 数组
        return np.array([data.embedding for data in response.data])
    except Exception as e:
        print(f"Batch Embedding Error: {e}")
        return np.zeros((len(texts), 4096))

def build_author_profiles(candidate_ids, author_db, whole_pub_db, target_paper: Dict):
    start_time = time.time()
    profiles_text = {}
    
    # 1. 获取目标论文向量
    target_text = f"{target_paper.get('title', '')} {' '.join(target_paper.get('keywords', []))}".strip()
    target_res = get_embeddings_batch([target_text])
    if target_res.size == 0: return {}
    target_embedding = target_res[0]

    # 2. 遍历候选人
    for auth_id in candidate_ids:
        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        all_orgs_normalized = []
        global_collaborators = Counter()
        pub_texts = []
        
        # 预处理该作者所有论文
        for pid in pub_ids:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            
            # 文本拼接
            t = f"{pub_detail.get('title','')} {' '.join(pub_detail.get('keywords',[]))}".strip()
            pub_texts.append(t if t else "n/a")

            # 机构与合作者统计
            for auth_entry in pub_detail.get('authors', []):
                if same_name(auth_entry.get('name', ''), basic_info.get('name', '')):
                    if auth_entry.get('org'):
                        norm_org = normalize_org(auth_entry.get('org'))
                        if norm_org: all_orgs_normalized.append(norm_org)
                else:
                    name = auth_entry.get('name')
                    if name: global_collaborators[name] += 1

        if not pub_texts:
            profiles_text[auth_id] = f"【 ID: {auth_id} 】\n(No publications found)"
            continue

        # --- 性能爆发点：一次性请求该作者的所有论文向量 ---
        cand_embs = get_embeddings_batch(pub_texts)
        
        # 向量化计算余弦相似度 (Vectorized Ops)
        norm_cand = np.linalg.norm(cand_embs, axis=1) + 1e-9
        norm_target = np.linalg.norm(target_embedding) + 1e-9
        scores = (cand_embs @ target_embedding) / (norm_cand * norm_target)
        
        # 获取最相关的 Top 5
        top_indices = np.argsort(scores)[-5:][::-1]

        # 格式化输出
        unique_orgs = list(dict.fromkeys(all_orgs_normalized)) # 去重且保持顺序
        top_collabs = global_collaborators.most_common(5)

        desc = f"【 ID: {auth_id} 】\n"
        desc += "- orgs:\n"
        if unique_orgs:
            for i, org in enumerate(unique_orgs[:5]):
                desc += f"  {i+1}. {org}\n"
        else:
            desc += "  (Unknown)\n"

        desc += "- keywords: "
        top_keywords = []
        for i in top_indices:
            words = re.findall(r"[a-zA-Z]{4,}", pub_texts[i].lower())
            top_keywords.extend(words)
        kw_counter = Counter(top_keywords)
        top_kws = [kw for kw, _ in kw_counter.most_common(10)]
        desc += (", ".join(top_kws) if top_kws else "N/A") + "\n"

        desc += "- top_works:\n"
        for i, idx in enumerate(top_indices):
            desc += f"  {i+1}. {pub_texts[idx][:120]}... (Score: {scores[idx]:.3f})\n"
            
        desc += "- collaborators: "
        desc += (", ".join([c[0] for c in top_collabs]) if top_collabs else "N/A") + "\n"
        
        profiles_text[auth_id] = desc

    end_time = time.time()
    print(f"Total processing time for {len(candidate_ids)} candidates: {end_time - start_time:.2f}s")
    return profiles_text