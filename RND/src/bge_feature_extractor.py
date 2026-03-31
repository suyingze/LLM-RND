# -*- coding: utf-8 -*-
import json
import os
import re
import torch
import numpy as np
import glob
import huggingface_hub
from collections import Counter
from typing import Dict, List
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file
VECTOR_CACHE_DIR = "output/vector_cache"
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['SENTENCE_TRANSFORMERS_OFFLINE'] = '1' 

snapshot_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*")
snapshot_paths = glob.glob(snapshot_pattern)
if snapshot_paths:
    snapshot_path = snapshot_paths[0]  # 取第一个找到的快照
    print(f"找到本地模型: {snapshot_path}")
    print(f"路径是否存在: {os.path.exists(snapshot_path)}")
else:
    print("未找到本地模型缓存，尝试从网络下载...")
    # 如果找不到，才从网络下载（临时关闭离线模式）
    os.environ.pop('HF_HUB_OFFLINE', None)
    snapshot_path = "BAAI/bge-m3"  
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(snapshot_path, device=device)


# 常见噪音词/层级词（用于文本清洗）

NOISE_PATTERNS = [
    r"school of", r"department of", r"dept of", r"dept\.",
    r"college of", r"institute of", r"lab of", r"laboratory of",
    r"faculty of", r"division of", r"center for", r"centre for"
]

# 语义核心关键词（用于优先识别真实机构）
CORE_KEYWORDS = [
    "university", "institute", "academy", "college", "center", "laboratory",
    "research", "school"
]

# 缩写/别名映射表（可扩展）
ABBR_MAP: Dict[str, str] = {
    "mit": "massachusetts institute of technology",
    "cas": "chinese academy of sciences",
    "ucas": "university of chinese academy of sciences",
    "caltech": "california institute of technology",
    "oxford": "university of oxford",
    # 持续补充
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

def merge_similar_orgs(org_list: List[str], threshold: int = 80) -> List[str]:
    merged = []
    for norm_org in org_list:
        if not norm_org: continue
        found = False
        for i, exist in enumerate(merged):
            if fuzz.ratio(norm_org.lower(), exist.lower()) >= threshold:
                if len(norm_org) > len(exist):
                    merged[i] = norm_org
                found = True
                break
        if not found:
            merged.append(norm_org)
    return merged

def normalize_name(name: str) -> str:
    """把名字统一成 'token_token' 的形式：全小写、去标点、空白归一"""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\.\,\-]+", " ", s)   
    s = re.sub(r"\s+", " ", s)         
    parts = [p for p in s.split(" ") if p]
    return "_".join(parts)

def same_name(a: str, b: str) -> bool:
    """允许名-姓顺序互换"""
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    pa = na.split("_")
    pb = nb.split("_")

    # 情况 1：两段名，顺序互换
    if len(pa) == 2 and len(pb) == 2:
        if pa[0] == pb[1] and pa[1] == pb[0]:
            return True

    # 情况 2：缩写名（j_li vs jian_li）
    if len(pa) == 2 and len(pb) == 2:
        # pa 是缩写
        if len(pa[0]) == 1 and pa[1] == pb[1] and pb[0].startswith(pa[0]):
            return True
        # pb 是缩写
        if len(pb[0]) == 1 and pa[1] == pb[1] and pa[0].startswith(pb[0]):
            return True
        
    # 情况 3: 多段名，首尾互换
    return False
    
@torch.no_grad()
def build_author_profiles(candidate_ids, author_db, whole_pub_db, target_paper: Dict):
    MODEL.max_seq_length = 256
    MODEL.half()
    profiles_text = {}

    target_text = f"{target_paper.get('title', '')} {' '.join(target_paper.get('keywords', []))}".strip()
    target_embedding = MODEL.encode(
        [target_text],
        batch_size=1,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).half()[0]
   
    for auth_id in candidate_ids:
        cache_path = os.path.join(VECTOR_CACHE_DIR, f"{auth_id}.safetensors")
        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])

        current_author_name = basic_info.get('name', '')

        if os.path.exists(cache_path):
            data = load_file(cache_path)
            cand_embeddings = data["embeddings"].to(device).half()
        else:
            pub_texts_all = []
            for pid in pub_ids:
                p = whole_pub_db.get(pid, {})
                pub_texts_all.append(f"{p.get('title','')} {' '.join(p.get('keywords',[]))}".strip())
            if not pub_texts_all: continue
            cand_embeddings = MODEL.encode(pub_texts_all, batch_size=16, convert_to_tensor=True,normalize_embeddings=True).half()

        scores = cand_embeddings @ target_embedding
        # 取 top-k 论文来动态构建机构和合作者信息，k 的值可以根据实际情况调整
        top_k_val = min(5, cand_embeddings.size(0))
        topk = torch.topk(scores, k=top_k_val)
        top_indices = topk.indices.tolist()

        # THRESHOLD = 0.67 # 阈值
        # MIN_KEEP = 3       # 搜索质量差时的保底数
        # MAX_KEEP = 10      # 搜索结果爆炸时的封顶数
        # mask = scores >= THRESHOLD
        # high_score_indices = torch.nonzero(mask).squeeze(-1)
        # # 获取按分数降序排列的所有索引
        # sorted_indices = torch.argsort(scores, descending=True)

        # if high_score_indices.numel() < MIN_KEEP:
        #     top_indices = sorted_indices[:min(MIN_KEEP, len(scores))].tolist()
        # else:
        #     top_indices = []
        #     for idx in sorted_indices:
        #         if scores[idx] >= THRESHOLD and len(top_indices) < MAX_KEEP:
        #             top_indices.append(idx.item())
        #         else:
        #             break

        dynamic_orgs = []
        dynamic_collabs = Counter()
        top_works_texts = []
        for idx in top_indices:
            pid = pub_ids[idx]
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            
            # 拼接论文文本用于关键词提取和 works 展示
            p_text = f"{pub_detail.get('title','')} {' '.join(pub_detail.get('keywords',[]))}".strip()
            top_works_texts.append(p_text)

            # 提取机构和合作者
            for auth_entry in pub_detail.get('authors', []):
                entry_name = auth_entry.get('name', '')
                if same_name(entry_name, current_author_name):
                    if auth_entry.get('org'):
                        norm_org = normalize_org(auth_entry.get('org'))
                        if norm_org: dynamic_orgs.append(norm_org)
                else:
                    if entry_name: dynamic_collabs[entry_name] += 1


        unique_orgs = list(dict.fromkeys(dynamic_orgs)) 
        top_collabs = dynamic_collabs.most_common(5)

        desc = f"【 ID: {auth_id} 】\n"
        desc += "- orgs:\n"
        if unique_orgs:
            for i, org in enumerate(unique_orgs[:5]):
                desc += f"  {i+1}. {org}\n" 
        else:
            desc += "  (Unknown)\n"

        desc += "- keywords: "
        top_keywords = []
        for text in top_works_texts:
            words = re.findall(r"[a-zA-Z]{4,}", text.lower())
            top_keywords.extend(words)
        kw_counter = Counter(top_keywords)
        top_kws = [kw for kw, _ in kw_counter.most_common(10)]
        desc += ", ".join(top_kws) if top_kws else "N/A"
        desc += "\n"
        desc += "- works:\n"
        for i, paper_text in enumerate(top_works_texts):
            desc += f"  {i+1}. {paper_text[:150]}\n"

        desc += "- collaborators: "
        if top_collabs:
            desc += ", ".join([c[0] for c in top_collabs])
        else:
            desc += "N/A"
        desc += "\n"

        profiles_text[auth_id] = desc
        del cand_embeddings

    return profiles_text