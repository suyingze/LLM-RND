# -*- coding: utf-8 -*-
import json
import os
import asyncio
import time
from src.sa_lzk.convert_gt import convert_to_snake_pinyin 
from src.candidate_generator import get_candidates # 内部逻辑需确保支持拼音匹配
from src.bge_feature_extractor import build_author_profiles
from src.llm_decider_sl import ask_deepseek_async
from src.llm_decider_twostage_sl import ask_deepseek_two_stage_async
from config import init_dspy 

# --- 1. 配置新路径 ---
os.environ["CURRENT_DATASET"] = "sa_lzk"
DATA_DIR = "dataset/sa_lzk_data"
UNASS_PATH = os.path.join(DATA_DIR, "unass.json")
PUB_DB_PATH = os.path.join(DATA_DIR, "pub.json") 
WHOLE_AUTHOR_PATH = os.path.join(DATA_DIR, "profiles", "whole_author_profiles.json")
WHOLE_PUB_PATH = os.path.join(DATA_DIR, "profiles", "whole_author_profiles_pub.json") 
GT_PATH = os.path.join(DATA_DIR, "cna_valid_ground_truth.json") 
OUTPUT_BASE = "output/sa_lzk" 
os.makedirs(OUTPUT_BASE, exist_ok=True)
SAVE_PATH = os.path.join(OUTPUT_BASE, "result.json")
LOG_PATH = os.path.join(OUTPUT_BASE, "analysis_log.jsonl")

sem = asyncio.Semaphore(5) 
file_lock = asyncio.Lock()
STRATEGY = 'HYBRID'

async def process_single_task(item, pubs_db, author_db, whole_pub_db, total_count, current_idx):
    """针对新数据集简化的异步工作流"""
    # 直接获取 wos 作为 task_id
    task_id = item.get('wos', 'unknown')
    target_name_cn = item.get('name', '')
    
    # 阶段 A: 粗筛
    # 构造 target_author 对象兼容旧接口，或者直接传 name
    target_name_key = convert_to_snake_pinyin(target_name_cn) 

    print(f"DEBUG: 原始姓名={target_name_cn} -> 检索Key={target_name_key}")

    target_author = {"name": target_name_key}
    candidate_ids = get_candidates(target_author, author_db)
    
    # 获取真实答案用于统计
    correct_auth_id = paper_to_author.get(task_id)
    
    if not candidate_ids:
        return task_id, "NIL", "No candidates", 0, 0, 0, 0, 0, 1, (correct_auth_id is None), target_name_key

    # 阶段 B: 特征提取
    # 这里的 item 本身就包含了题目的 title(lzmc), venue(cbsorqkmc) 等信息
    # 映射字段名以适配 bge_feature_extractor
    paper_info = {
        "title": item.get("lzmc", ""),
        "venue": item.get("cbsorqkmc", ""),
        "abstract": "" # 新数据中若无摘要则留空
    }
    
    candidate_profiles = build_author_profiles(candidate_ids, author_db, whole_pub_db, target_paper=paper_info)
    num_candidates = len(candidate_profiles)

    # 阶段 C: LLM 决策
    try:
        async with sem:
            if STRATEGY == 'HYBRID' and num_candidates > 20:
                return await ask_deepseek_two_stage_async(
                    task_id, paper_info, candidate_profiles, 
                    current_index=current_idx, 
                    target_name=target_name_cn,
                    gt_id=correct_auth_id,
                    total_count=total_count, 
                    target_name_key=target_name_key
                )
            else:
                target_id, reason, cand_count, in_t, out_t = await ask_deepseek_async(
                    task_id, paper_info, candidate_profiles, target_name_cn
                )
                return (task_id, target_id, reason, cand_count, cand_count, in_t, in_t, out_t, 1, (correct_auth_id is None), target_name_key)
    except Exception as e:
        return task_id, None, f"Error: {str(e)}", 0, 0, 0, 0, 0, 0, (correct_auth_id is None),target_name_key

async def main():
    start_time = time.perf_counter()
    init_dspy()

    print("正在加载 sa_lzk_data 数据库...")
    with open(UNASS_PATH, 'r', encoding='utf-8') as f: unass_list = json.load(f)
    with open(WHOLE_AUTHOR_PATH, 'r', encoding='utf-8') as f: author_db = json.load(f)
    with open(WHOLE_PUB_PATH, 'r', encoding='utf-8') as f: whole_pub_db = json.load(f)
    
    # 构建 GT 映射
    global paper_to_author
    paper_to_author = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            for name, authors in gt_data.items():
                for aid, pids in authors.items():
                    for pid in pids: paper_to_author[pid] = aid

    
    processed_tasks = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    full_tid = json.loads(line).get("task_id")
                    if full_tid:
                     
                        short_tid = full_tid.split('-')[0]
                        processed_tasks.add(short_tid)
                except: continue

    test_limit = 200 # 测试前100题
    tasks_to_run = [item for item in unass_list[:test_limit] if item.get('wos') not in processed_tasks]
    
    processed_count = len(processed_tasks)
    print(f" 进度统计：总测试量 {test_limit} | 已完成 {len(processed_tasks)} | 待处理 {len(tasks_to_run)}")
    current_global_idx = processed_count
    if not tasks_to_run:
        print(" 所有任务已在断点记录中，无需重跑。")
        return

    total_l1_hits = 0
    total_actual_run = 0
    actual_nil_count = 0

    # 分批处理
    BATCH_SIZE = 50
    results = {}
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f" 成功加载已有结果，当前包含 {len(results)} 个作者的记录。")
        except Exception as e:
            print(f" 加载旧结果失败，将从空开始: {e}")
            results = {}
    for i in range(0, len(tasks_to_run), BATCH_SIZE):
        batch = tasks_to_run[i : i + BATCH_SIZE]
        coros = [
            process_single_task(item, None, author_db, whole_pub_db, test_limit, len(processed_tasks) + i + idx + 1) 
            for idx, item in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*coros)

       

        async with file_lock:
               for tid, target_id, reason, l1_c, l2_c, ts_in, orig_in, out_t, l1_hit, is_nil_case, formatted_name in batch_results:
                task_id_with_name = f"{tid}-{formatted_name}"
                total_actual_run += 1
                total_l1_hits += l1_hit
                if is_nil_case:
                    actual_nil_count += 1

                final_res = target_id if target_id else "NIL"
                analysis_entry = {
                    "task_id": task_id_with_name,
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
                if task_id_with_name not in results[key]:
                    results[key].append(task_id_with_name)

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    if total_actual_run > 0:
        overall_hit_rate = (total_l1_hits / total_actual_run) * 100
        id_cases = total_actual_run - actual_nil_count

        end_time = time.perf_counter()
        total_time = end_time - start_time

        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

    
    
    
        print("\n" + "="*50)
        print(f" L1 阶段命中率汇总 (本次运行):")
        print(f"   - 实际处理总数: {total_actual_run}")
        print(f"   - L1 命中总数: {total_l1_hits} (含NIL默认命中)")
        print(f"   - 其中 NIL 样本数: {actual_nil_count}")
        print(f"   - 其中 已有作者(ID)样本数: {id_cases}")
        print(f"   - 总体命中率: {overall_hit_rate:.2f}%")
        print(f"   - 总运行时间: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        print("="*50 + "\n")
    print(f"\n处理完成！")


if __name__ == "__main__":
    asyncio.run(main())