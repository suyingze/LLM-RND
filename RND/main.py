# -*- coding: utf-8 -*-
import json
import os
import random
from src.candidate_generator import get_target_author, get_candidates
from src.feature_extractor import build_author_profiles 
from src.llm_decider import ask_deepseek
from config import init_dspy 

# 配置路径 
DATA_DIR = "dataset/valid"
UNASS_PATH = os.path.join(DATA_DIR, "cna_valid_unass.json")
UNASS_PUB_PATH = os.path.join(DATA_DIR, "cna_valid_unass_pub.json")
WHOLE_AUTHOR_PATH = os.path.join(DATA_DIR, "whole_author_profiles.json")
WHOLE_PUB_PATH = os.path.join(DATA_DIR, "whole_author_profiles_pub.json") 
SAVE_PATH = "output/result.json"
LOG_PATH = "output/analysis_log.jsonl"
def main():
    # 初始化 DSPy
    init_dspy()

    # 1. 初始化数据加载
    print("正在加载题目集与全量数据库，请稍候...")
    with open(UNASS_PATH, 'r', encoding='utf-8') as f:
        unass_list = json.load(f)
    with open(UNASS_PUB_PATH, 'r', encoding='utf-8') as f:
        pubs_db = json.load(f)
    with open(WHOLE_AUTHOR_PATH, 'r', encoding='utf-8') as f:
        author_db = json.load(f)
    with open(WHOLE_PUB_PATH, 'r', encoding='utf-8') as f:
        whole_pub_db = json.load(f)
    #print(f"数据加载完成！共有 {len(unass_list)} 篇待处理论文。")

    # 2. 断点恢复与任务准备
    results = {}
    processed_tasks = set()

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "task_id" in entry and "stats" in entry and entry.get("result"):
                        tid = entry["task_id"]
                        processed_tasks.add(tid)
                        res_id = entry["result"] if entry["result"] != "NIL" else "new_author"
                        results.setdefault(res_id, []).append(tid)
                except: continue

    random_tasks = random.sample(unass_list, 2)
    tasks_to_run = [t for t in random_tasks if t not in processed_tasks]

    num_skipped = len(processed_tasks)
    total_all = len(random_tasks)
    print(f"检测到断点：已跳过 {num_skipped} 条。还剩 {len(tasks_to_run)} 条需处理。")

    # 3. 逐条处理
    #test_limit = 100 # 测试前100个
    # for i, task_id in enumerate(unass_list[:test_limit]):
    for i, task_id in enumerate(tasks_to_run):
        current_global_idx = i + 1 + num_skipped

        paper_id, author_idx = task_id.split('-')
        author_idx = int(author_idx)

        # 阶段 A: 粗筛候选人
        paper_info = pubs_db.get(paper_id, {})
        target_author = get_target_author(paper_info, author_idx)
        candidate_ids = get_candidates(target_author, author_db)

        if not candidate_ids:
            print(f"[{current_global_idx}/{total_all}] 任务 {task_id} 无候选人 -> 自动记录为 NIL")
            analysis_entry = {
                "task_id": task_id,
                "stats": {"candidates": 0, "input_tokens": 0, "output_tokens": 0},
                "result": "NIL",
                "reasoning": "Candidate set is empty."
            }
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(analysis_entry, ensure_ascii=False) + "\n")
                f.flush()
            
            results.setdefault("new_author", []).append(task_id)
            processed_tasks.add(task_id)
            continue

        # 阶段 B: 特征提取 
        candidate_profiles = build_author_profiles(candidate_ids, author_db, whole_pub_db)
       
        # 阶段 C: LLM 决策 
        try:
            target_id, reason, cand_count, in_t, out_t = ask_deepseek(
                task_id, paper_info, candidate_profiles, 
                current_index=current_global_idx, 
                total_count=total_all
            )

            analysis_entry = {
                "task_id": task_id,
                "stats": {"candidates": cand_count, "input_tokens": in_t, "output_tokens": out_t},
                "result": target_id if target_id else "NIL",
                "reasoning": reason
            }

            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(analysis_entry, ensure_ascii=False) + "\n")
                f.flush()

            # 更新结果并保存
            final_key = target_id if target_id else "new_author"
            results.setdefault(final_key, []).append(task_id)
            processed_tasks.add(task_id)

            if (i + 1) % 1 == 0:
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"任务 {task_id} 出错: {e}")
            continue

    print(f"\n消歧结束，结果已保存至 {SAVE_PATH}")

if __name__ == "__main__":
    main()
