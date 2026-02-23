# -*- coding: utf-8 -*-
import json
import os
import random
import asyncio
from src.candidate_generator import get_target_author, get_candidates
from src.full_feature_extractor import build_author_profiles 
from src.llm_decider import ask_deepseek_async
from src.llm_decider_twostage import ask_deepseek_two_stage_async
from config import init_dspy 

# 配置路径 
DATA_DIR = "dataset/valid"
UNASS_PATH = os.path.join(DATA_DIR, "cna_valid_unass.json")
UNASS_PUB_PATH = os.path.join(DATA_DIR, "cna_valid_unass_pub.json")
WHOLE_AUTHOR_PATH = os.path.join(DATA_DIR, "whole_author_profiles.json")
WHOLE_PUB_PATH = os.path.join(DATA_DIR, "whole_author_profiles_pub.json") 
SAVE_PATH = "output/result.json"
LOG_PATH = "output/analysis_log.jsonl"
# 并发控制锁和信号量
file_lock = asyncio.Lock()
sem = asyncio.Semaphore(2)  # 限制同时开启 9 个 LLM 请求 HYBRID模式/SINGLE建议 3
# 'SINGLE' - 全部强制走单层（用于跑 Baseline 数据）
# 'HYBRID' - 混合模式：候选人 > 20 走两层，否则走单层
STRATEGY = 'HYBRID'

async def process_single_task(task_id, pubs_db, author_db, whole_pub_db, results, total_count, current_idx):
    """单个任务的异步工作流"""
    async with sem:
        paper_id, author_idx = task_id.split('-')
        author_idx = int(author_idx)

        # 阶段 A: 粗筛 (本地计算)
        paper_info = pubs_db.get(paper_id, {})
        target_author = get_target_author(paper_info, author_idx)
        target_name = target_author.get('name', "")
        candidate_ids = get_candidates(target_author, author_db)
        correct_auth_id = paper_to_author.get(paper_id)
        if not candidate_ids:
            return task_id, "NIL", "No candidates", 0, 0, 0, 0, 0, 1, (correct_auth_id is None)

        # 阶段 B: 特征提取 (带磁盘缓存)
        candidate_profiles = build_author_profiles(candidate_ids, author_db, whole_pub_db)
        num_candidates = len(candidate_profiles)
        # 阶段 C: LLM 决策 (异步 I/O)
        try:
            paper_id = task_id.split('-')[0]
            correct_auth_id = paper_to_author.get(paper_id)

            if STRATEGY == 'HYBRID' and num_candidates > 20:
               return await ask_deepseek_two_stage_async(
                  task_id, paper_info, candidate_profiles, 
                  current_index=current_idx, 
                  target_name=target_name,
                  gt_id=correct_auth_id,
                   total_count=total_count
                )
            else:
                target_id, reason, cand_count, in_t, out_t = await ask_deepseek_async(
                    task_id, paper_info, candidate_profiles, target_name
                )
                l1_hit_dummy = 1 
                is_nil_dummy = (correct_auth_id is None)
                return (task_id, target_id, reason, cand_count, cand_count, in_t, in_t, out_t, l1_hit_dummy, is_nil_dummy)

        except Exception as e:
            print(f" 任务 {task_id} LLM 调用失败: {e}")
            return task_id, None, f"Error: {str(e)}", 0, 0, 0, 0, 0, 0, (correct_auth_id is None)
async def main():
    init_dspy()

    # 1. 加载数据
    print("正在加载数据库...")
    with open(UNASS_PATH, 'r', encoding='utf-8') as f: unass_list = json.load(f)
    with open(UNASS_PUB_PATH, 'r', encoding='utf-8') as f: pubs_db = json.load(f)
    with open(WHOLE_AUTHOR_PATH, 'r', encoding='utf-8') as f: author_db = json.load(f)
    with open(WHOLE_PUB_PATH, 'r', encoding='utf-8') as f: whole_pub_db = json.load(f)
    GT_PATH = os.path.join(DATA_DIR, "cna_valid_ground_truth.json")
    global paper_to_author
    paper_to_author = {}
    with open(GT_PATH, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
            # 构建 论文ID -> 作者ID 的映射
            for name, authors in ground_truth.items():
                for auth_id, papers in authors.items():
                    for pid in papers:
                        paper_to_author[pid] = auth_id

    # 2. 精确断点恢复
    results = {}
    processed_tasks = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    tid = entry.get("task_id")
                    if tid:
                        processed_tasks.add(tid)
                        res_id = entry.get("result", "NIL")
                        results.setdefault(res_id, []).append(tid)
                except: continue

    test_limit = 40
    candidate_pool = unass_list[:test_limit]
    # 只处理不在 processed_tasks 里的任务
    tasks_to_run = [t for t in candidate_pool if t not in processed_tasks]
    
    print(f" 进度统计：总测试量 {test_limit} | 已完成 {len(processed_tasks)} | 待处理 {len(tasks_to_run)}")

    if not tasks_to_run:
        print(" 所有任务已在断点记录中，无需重跑。")
        return

    total_l1_hits = 0
    total_actual_run = 0
    actual_nil_count = 0

    # 3. 分批异步处理 (Batch Processing)
    BATCH_SIZE = 2
    for i in range(0, len(tasks_to_run), BATCH_SIZE):
        batch = tasks_to_run[i : i + BATCH_SIZE]
        coros = [
            process_single_task(tid, pubs_db, author_db, whole_pub_db, results, test_limit, len(processed_tasks) + i + idx + 1) 
            for idx, tid in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*coros)

        # 4. 批量保存结果
        async with file_lock:
            for tid, target_id, reason, l1_c, l2_c, ts_in, orig_in, out_t, l1_hit, is_nil_case in batch_results:
                total_actual_run += 1
                total_l1_hits += l1_hit
                if is_nil_case:
                    actual_nil_count += 1

                final_res = target_id if target_id else "NIL"
                analysis_entry = {
                    "task_id": tid,
                    "stats": {
                        "l1_hit": "YES" if l1_hit == 1 else "NO",
                        "is_new_author": is_nil_case,
                        "candidates_ratio": f"{l1_c} -> {l2_c}",
                        "input_tokens_comparison": {
                        "original_single_layer": orig_in,    # 原单层全量输入
                        "two_layer_total": ts_in,            # 两层合计输入 (L1+L2)
                        "saved_tokens": orig_in - ts_in      # 节省的 Token
                    },
                    "output_tokens": out_t
                    },
                    "result": final_res,
                    "reasoning": reason
                }
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(analysis_entry, ensure_ascii=False) + "\n")
                
                #  更新内存中的字典 (使用 NIL 或具体 ID)
                key = final_res if final_res != "NIL" else "new_author"
                if key not in results:
                    results[key] = []
                if tid not in results[key]:
                    results[key].append(tid)

            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    if total_actual_run > 0:
        overall_hit_rate = (total_l1_hits / total_actual_run) * 100
        id_cases = total_actual_run - actual_nil_count
        
        print("\n" + "="*50)
        print(f" L1 阶段命中率汇总 (本次运行):")
        print(f"   - 实际处理总数: {total_actual_run}")
        print(f"   - L1 命中总数: {total_l1_hits} (含NIL默认命中)")
        print(f"   - 其中 NIL 样本数: {actual_nil_count}")
        print(f"   - 其中 已有作者(ID)样本数: {id_cases}")
        print(f"   - 总体命中率: {overall_hit_rate:.2f}%")
        print("="*50 + "\n")
    print(f"\n处理完成！")

if __name__ == "__main__":
    asyncio.run(main())