# -*- coding: utf-8 -*-
import json
import os
import re
import torch
import numpy as np
from collections import Counter
from typing import Dict, List
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("BAAI/bge-m3", device=device)# 1024维语义向量模型，专门针对中文优化，适合做文本相似度计算和语义搜索
print("Using device:", device)

CACHE_FILE = "output/profile_cache.json"

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
    
def build_author_profiles(candidate_ids, author_db, whole_pub_db, target_paper: Dict):
    profiles_text = {}
    # 预处理待消歧论文的特征向量
    target_title = target_paper.get('title', '')
    target_kws = " ".join(target_paper.get('keywords', []))
    target_text = f"{target_title} {target_kws}".strip()


    target_embedding = MODEL.encode(
        [target_text],
        batch_size=16,
        convert_to_tensor=True,
        normalize_embeddings=True
    )[0]
    author_data = {}      # 存储每个作者的统计信息
    all_texts = []        # 所有候选论文文本
    text_owner = []       # 每条文本属于哪个作者

    #print("target_embedding shape:", target_embedding.shape)
    
    for auth_id in candidate_ids:
        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        all_orgs_normalized = []
        global_collaborators = Counter()
        pub_texts = []

        #  收集该候选人名下的所有论文详情
        for pid in pub_ids:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            
            text = f"{pub_detail.get('title','')} {' '.join(pub_detail.get('keywords',[]))}".strip()
            if text:
                pub_texts.append(text)
                all_texts.append(text)
                text_owner.append(auth_id)

            # 统计机构 (全局)
            for auth_entry in pub_detail.get('authors', []):
                if same_name(auth_entry.get('name', ''), basic_info.get('name', '')):
                    if auth_entry.get('org'):
                        norm_org = normalize_org(auth_entry.get('org'))
                        if norm_org: all_orgs_normalized.append(norm_org)
                else:
                    name = auth_entry.get('name')
                    if name: global_collaborators[name] += 1

        author_data[auth_id] = {
            "orgs": all_orgs_normalized,
            "collabs": global_collaborators,
            "pub_count": len(pub_texts)
        }

    if len(all_texts) > 0:
        all_embeddings = MODEL.encode(
            all_texts,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    else:
        all_embeddings = []
        print(" No publication texts found!")

    idx = 0

    for auth_id in candidate_ids:

        pub_count = author_data[auth_id]["pub_count"]

        if pub_count == 0:
            profiles_text[auth_id] = f"【 ID: {auth_id} 】\n(No publications found)"
            continue

        cand_embeddings = all_embeddings[idx: idx + pub_count]
        idx += pub_count


        scores = cand_embeddings @ target_embedding

        topk = torch.topk(scores, k=min(5, pub_count))

        unique_orgs = list(set(author_data[auth_id]["orgs"]))
        top_collabs = author_data[auth_id]["collabs"].most_common(5)

        desc = f"【 ID: {auth_id} 】\n"
        desc += "- orgs:\n"
        if unique_orgs:
            for i, org in enumerate(unique_orgs[:5]):
                desc += f"  {i+1}. {org}\n"
        else:
            desc += "  (Unknown)\n"

        desc += "- keywords: "
        top_keywords = []
        for i in topk.indices.tolist():
            text = all_texts[idx - pub_count + i]
            words = re.findall(r"[a-zA-Z]{4,}", text.lower())
            top_keywords.extend(words)
        kw_counter = Counter(top_keywords)
        top_kws = [kw for kw, _ in kw_counter.most_common(10)]
        desc += ", ".join(top_kws) if top_kws else "N/A"
        desc += "\n"
        
        desc += "- works:\n"
        for i, rel_idx in enumerate(topk.indices.tolist()):
            paper_text = all_texts[idx - pub_count + rel_idx]
            desc += f"  {i+1}. {paper_text[:150]}\n"
        desc += "- collaborators: "
        if top_collabs:
            desc += ", ".join([c[0] for c in top_collabs])
        else:
            desc += "N/A"
        desc += "\n"

        profiles_text[auth_id] = desc
        

    return profiles_text

