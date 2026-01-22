# -*- coding: utf-8 -*-
from collections import Counter

import re

def build_author_profiles(candidate_ids, author_db, whole_pub_db):
    """
    特征提取模块：将候选人的背景数据转化为自然语言画像
    """
    profiles_text = {}

    for auth_id in candidate_ids:
        # 1. 获取候选作者们在数据库中的基本信息和论文列表
        basic_info = author_db.get(auth_id, {})
        pub_ids = basic_info.get('pubs', [])
        
        all_orgs = set()
        all_collaborators = Counter()
        titles = []
        keywords_pool = Counter()

        # 2. 深入全量论文库获取详情 (取前10篇论文)
        for pid in pub_ids[:10]:
            pub_detail = whole_pub_db.get(pid)
            if not pub_detail:
                continue
            
            # 提取标题
            titles.append(pub_detail.get('title', ''))
            
            # 提取关键词
            kws = pub_detail.get('keywords', [])
            keywords_pool.update(kws[:5])
            
            # 提取机构和合作者
            # 在全量论文详情中，我们需要找到这个候选人对应的那个条目
            for auth_entry in pub_detail.get('authors', []):
                # 如果名字匹配（归一化对比）
                if same_name(auth_entry.get('name', ''), basic_info.get('name', '')):
                    if auth_entry.get('org'):
                        org = normalize_org(auth_entry.get('org'))
                        if org:
                            all_orgs.add(org)

                else:
                    # 其他人就是合作者，并记录频率
                    name = auth_entry.get('name')
                    if name:
                         all_collaborators[name] += 1

        # 3. 组织成自然语言描述
        desc = f"【候选人 ID: {auth_id}】\n"
        org_list = sorted(all_orgs)
        desc += f"- 历史就职机构: {'; '.join(org_list[:5]) if org_list else '未知'}\n"
        top_keywords = [kw for kw, _ in keywords_pool.most_common(8)]
        desc += f"- 核心研究主题: {', '.join(top_keywords)}\n"
        desc += f"- 代表性论文标题: {'; '.join(titles[:3])} 等\n"
        top_collaborators = [name for name, _ in all_collaborators.most_common(10)]
        desc += f"- 主要合作者: {', '.join(top_collaborators)}\n"

        

        profiles_text[auth_id] = desc

    return profiles_text

def normalize_org(org: str) -> str:
    if not org:
        return ""
    s = org.strip()
    s = re.sub(r"\s+", " ", s)              # 多空格合一
    s = s.rstrip(",;")                      # 去掉末尾逗号分号
    s = re.sub(r"\(([^()]*)\)", r"(\1)", s) # 保持括号格式稳定
    return s

def normalize_name(name: str) -> str:
    """把名字统一成 'token_token' 的形式：全小写、去标点、空白归一"""
    if not name:
        return ""
    s = name.strip().lower()
    # 把常见分隔符统一成空格
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
    return False
    return profiles_text
    
    # 情况 3: 多段名，首尾互换

