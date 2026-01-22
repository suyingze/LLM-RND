import json
import os
# -*- coding: utf-8 -*-
from src.feature_extractor import same_name

def get_target_author(paper_info, author_idx):
    """
    根据后缀索引获取具体的待消歧作者对象 
    """
    authors = paper_info.get('authors', [])
    if author_idx < len(authors):
        return authors[author_idx]
    return None

def get_candidates(target_author, author_db):
    """
    第一阶段：召回。根据名字找到所有可能的专家 ID。
    """
    target_name = target_author.get('name', "")
    candidates = []
    
    for author_id, profile in author_db.items():
        profile_name = profile.get('name', "")
        if same_name(profile_name, target_name):
            candidates.append(author_id)
            
    return candidates
