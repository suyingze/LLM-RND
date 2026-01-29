# -*- coding: utf-8 -*-
import json
import os
from src.candidate_generator import get_target_author, get_candidates
from src.feature_extractor import build_author_profiles

DEBUG_DIR = "output/debug_view"
os.makedirs(DEBUG_DIR, exist_ok=True)

def visualize_task(task_id, pubs_db, author_db, whole_pub_db):
    """
    生成一个易读的文本文件，展示最终输入给 LLM 的上下文。
    """
    paper_id, author_idx = task_id.split('-')
    author_idx = int(author_idx)
    
    paper_info = pubs_db.get(paper_id, {})
    target_author = get_target_author(paper_info, author_idx)
    candidate_ids = get_candidates(target_author, author_db)
    
    candidate_profiles = build_author_profiles(candidate_ids, author_db, whole_pub_db)

    output = []
    output.append(f" [调试任务 ID]: {task_id}")
    output.append(f"[待消歧姓名]: {target_author} (索引: {author_idx})")
    output.append("=" * 60)
    
    output.append("\n【PART 1: 待分配的论文详情 】")
    output.append(f"标题: {paper_info.get('title')}")
    output.append(f"机构: {paper_info.get('authors', [])[author_idx].get('org', '未知')}")
    output.append(f"关键词: {', '.join(paper_info.get('keywords', []))}")
    output.append(f"合作者: {', '.join([a.get('name') for a in paper_info.get('authors', [])])}")
    output.append(f"摘要: {paper_info.get('abstract', 'N/A')}")
    
    output.append("\n" + "=" * 60)
    output.append(f"【PART 2: 候选人画像 (Candidates - 共 {len(candidate_profiles)} 个)】")
    
    if not candidate_profiles:
        output.append(">>> 结果: NIL (没有找到候选人)")
    else:
        for auth_id, profile in candidate_profiles.items():
            output.append("-" * 40)
            output.append(profile)

    file_path = os.path.join(DEBUG_DIR, f"{task_id}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    
    print(f" 可视化完成！请查看: {file_path}")

if __name__ == "__main__":
    print("正在加载数据库以便可视化...")
    with open("dataset/valid/cna_valid_unass_pub.json", 'r', encoding='utf-8') as f: p_db = json.load(f)
    with open("dataset/valid/whole_author_profiles.json", 'r', encoding='utf-8') as f: a_db = json.load(f)
    with open("dataset/valid/whole_author_profiles_pub.json", 'r', encoding='utf-8') as f: w_db = json.load(f)

    test_tasks = ["E1B57TKe-0 "] 
    for tid in test_tasks:
        visualize_task(tid, p_db, a_db, w_db)