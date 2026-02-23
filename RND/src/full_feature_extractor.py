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

def build_author_profiles(candidate_ids, author_db, whole_pub_db):
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
        if auth_id in full_cache:
            profiles_text[auth_id] = full_cache[auth_id]
            continue

        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        all_orgs_normalized = []
        all_collaborators = Counter()
        titles_with_meta = []
        keywords_pool = Counter()

        for pid in pub_ids:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail: continue
            
            raw_title = pub_detail.get('title', 'Unknown Title')
            year = pub_detail.get('year', 'N/A')
            venue = pub_detail.get('venue', 'Unknown Venue')
            # 格式化为：标题 (Year | Venue)
            full_title_meta = f"{raw_title} (Year: {year} | Venue: {venue})"
            titles_with_meta.append(full_title_meta)
            
            keywords_pool.update(pub_detail.get('keywords', []))
            
            for auth_entry in pub_detail.get('authors', []):
                if same_name(auth_entry.get('name', ''), basic_info.get('name', '')):
                    if auth_entry.get('org'):
                        norm_org = normalize_org(auth_entry.get('org'))
                        if norm_org: all_orgs_normalized.append(norm_org)
                else:
                    name = auth_entry.get('name')
                    if name: all_collaborators[name] += 1

        unique_orgs = merge_similar_orgs(all_orgs_normalized)
        
        desc = f"【 ID: {auth_id}】\n"
        desc += "- orgs:\n"
        if unique_orgs:
            for i, org in enumerate(unique_orgs[:8]): 
                desc += f"  {i+1}. {org}\n"
        else:
            desc += "  (Unknown/Not provided)\n"
        top_kws = [f"{kw}({c}次)" for kw, c in keywords_pool.most_common(30)]
        desc += f"- keywords: {', '.join(top_kws)}\n"

        desc += "- works:\n"
        for i, t_meta in enumerate(titles_with_meta[:10]): 
            desc += f"  {i+1}. {t_meta}\n"
            
        top_cols = [f"{n}({c}次)" for n, c in all_collaborators.most_common(20)]
        desc += f"- collaborators: {', '.join(top_cols)}\n"
        
        full_cache[auth_id] = desc
        profiles_text[auth_id] = desc
        new_extracted_count += 1

    if new_extracted_count > 0:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(full_cache, f, ensure_ascii=False, indent=4)
            
    return profiles_text


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
    
    

