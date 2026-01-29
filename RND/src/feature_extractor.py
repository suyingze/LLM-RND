# -*- coding: utf-8 -*-
import json
import os
import re
from collections import Counter
from typing import Dict, List
from rapidfuzz import fuzz

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
    """org_list 此时应该是已经经过 normalize_org 处理后的列表"""
    merged = []
    for norm_org in org_list:
        if not norm_org: continue
        found = False
        for i, exist in enumerate(merged):
            if fuzz.ratio(norm_org.lower(), exist.lower()) >= threshold:
                # 保留较长的名称作为最终显示
                if len(norm_org) > len(exist):
                    merged[i] = norm_org
                found = True
                break
        if not found:
            merged.append(norm_org)
    return merged

def build_author_profiles(candidate_ids, author_db, whole_pub_db):
    # 1. 稳健的缓存加载逻辑
    full_cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                full_cache = json.load(f)
        except:
            full_cache = {}

    profiles_text = {}
    new_extracted_count = 0

    for auth_id in candidate_ids:
        # 2. 检查缓存
        if auth_id in full_cache:
            profiles_text[auth_id] = full_cache[auth_id]
            continue

        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        all_orgs_normalized = []
        all_collaborators = Counter()
        titles = []
        keywords_pool = Counter()

        for pid in pub_ids[:10]:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            
            # 标题压缩：只取前 10 个词，防止过长
            t = pub_detail.get('title', '')
            titles.append(" ".join(t.split()[:10]))
            
            keywords_pool.update(pub_detail.get('keywords', [])[:5])
            
            for auth_entry in pub_detail.get('authors', []):
                if same_name(auth_entry.get('name', ''), basic_info.get('name', '')):
                    if auth_entry.get('org'):
                        # 先清洗，再存入待合并列表
                        norm_org = normalize_org(auth_entry.get('org'))
                        if norm_org: all_orgs_normalized.append(norm_org)
                else:
                    name = auth_entry.get('name')
                    if name: all_collaborators[name] += 1

        # 3. 组织精简描述
        # 使用模糊匹配合并相似机构
        unique_orgs = merge_similar_orgs(all_orgs_normalized)
        
        desc = f"【候选人 ID: {auth_id}】\n"
        desc += f"- 机构: {'; '.join(unique_orgs[:3]) if unique_orgs else '未知'}\n"
        top_kws = [f"{kw}({c}次)" for kw, c in keywords_pool.most_common(6)]
        desc += f"- 核心主题: {', '.join(top_kws)}\n"
        desc += f"- 代表论文: {'; '.join(titles[:2])} 等\n"
        top_cols = [f"{n}({c}次)" for n, c in all_collaborators.most_common(8)]
        desc += f"- 核心合作者: {', '.join(top_cols)}\n"
        
        # 存入全局缓存
        full_cache[auth_id] = desc
        profiles_text[auth_id] = desc
        new_extracted_count += 1

    # 4. 只有在真正增加新数据时才保存，且保存的是全量 full_cache
    if new_extracted_count > 0:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(full_cache, f, ensure_ascii=False, indent=4)
            
    return profiles_text

# ... normalize_name 和 same_name 保持不变 ...

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
    
    

