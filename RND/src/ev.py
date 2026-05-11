# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict

def run_evaluation(pred_path, gt_path, is_test_mode=True):
    # 1. 加载数据
    if not os.path.exists(pred_path):
        print(f"错误：找不到预测文件 {pred_path}")
        return
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)  # {pred_auth_id: [task_id, ...]}
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f) # {name: {auth_id: [paper_id, ...]}}

    # 2. 建立全量真值索引
    paper_to_author_map = {}
    for name, authors in ground_truth.items():
        # 终极清洗：去掉所有下划线、空格，并转小写
        clean_name = name.replace("_", "").replace(" ", "").lower() 
        for auth_id, papers in authors.items():
            for pid in papers:
                paper_to_author_map[(clean_name, pid)] = auth_id 
    
    # 3. 整理预测结果
    author_pred_papers = defaultdict(set)
    all_task_papers_info = set() # 存储 (name, pid)
    all_pids = set()             # 新增：存储纯 pid 集合用于后续计算

    for pred_auth_id, task_ids in predictions.items():
        for task_id in task_ids:
            parts = task_id.split('-')
            pid = parts[0].strip()
            raw_name = parts[1].strip() if len(parts) > 1 else "unknown"
            # 这里的清洗逻辑必须和上面完全一致
            clean_name = raw_name.replace("_", "").replace(" ", "").lower()
        
            author_pred_papers[pred_auth_id].add(pid)
            all_task_papers_info.add((clean_name, pid))
            all_pids.add(pid)

    # 4. 根据 (name, pid) 确定真值作者
    active_true_authors = set()
    author_true_papers = defaultdict(set)

    for name, pid in all_task_papers_info:
        correct_auth_id = paper_to_author_map.get((name, pid))
    
        if correct_auth_id is not None:
            active_true_authors.add(correct_auth_id)
            author_true_papers[correct_auth_id].add(pid)
        else:
            # 标记为新作者 (NIL)
            active_true_authors.add("new_author")
            author_true_papers["new_author"].add(pid)

    # 5. 指标统计
    all_author_ids = active_true_authors 
    TUP = len(all_task_papers_info) # 总任务数
    
    WeightedPrecision = 0
    WeightedRecall = 0

    print(f"{'Author ID':<25} | {'CPA':<5} | {'TPA':<5} | {'UPA':<5} | {'P':<8} | {'R':<8}")
    print("-" * 85)

    for auth_id in all_author_ids:
        true_set = author_true_papers[auth_id]
        pred_set = author_pred_papers.get(auth_id, set()) # 使用 .get 防止 Key 缺失
        
        CPA = len(true_set & pred_set)
        TPA = len(pred_set)
        UPA = len(true_set)
        
        Precision_i = CPA / TPA if TPA > 0 else 0
        Recall_i = CPA / UPA if UPA > 0 else 0
        Weight_i = UPA / TUP
        
        WeightedPrecision += Precision_i * Weight_i
        WeightedRecall += Recall_i * Weight_i
        
        display_id = "NIL (new_author)" if auth_id == "new_author" else auth_id
        print(f"{str(display_id):<25} | {CPA:<5} | {TPA:<5} | {UPA:<5} | {Precision_i:<8.1%} | {Recall_i:<8.1%}")

    # 6. 计算最终 WeightedF1
    if (WeightedPrecision + WeightedRecall) > 0:
        WeightedF1 = 2 * (WeightedPrecision * WeightedRecall) / (WeightedPrecision + WeightedRecall)
    else:
        WeightedF1 = 0

    # 7. NIL 监控指标
    nil_true = author_true_papers.get("new_author", set())
    nil_pred = author_pred_papers.get("new_author", set())
    nil_recall = len(nil_true & nil_pred) / len(nil_true) if len(nil_true) > 0 else 0
    
    # 修正变量名：使用 all_pids
    old_author_papers_count = len(all_pids - nil_true)
    nil_false_positives = len(nil_pred - nil_true)
    fpr = nil_false_positives / old_author_papers_count if old_author_papers_count > 0 else 0

    print("="*85)
    print(f" 评估报告 (M = {len(all_author_ids)}, TUP = {TUP}):")
    print(f"  > WeightedPrecision: {WeightedPrecision:.4f}")
    print(f"  > WeightedRecall:    {WeightedRecall:.4f}")
    print(f"  > WeightedF1 Score:  {WeightedF1:.4f}")
    print("-" * 85)
    print(f"  >  NIL召回率: {nil_recall:.2%} | NIL误报率(FPR): {fpr:.2%}")
    print("="*85)

if __name__ == "__main__":
    # 路径根据你的实际环境调整
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRED_FILE = os.path.join(BASE_DIR, "output", "sa_lzk", "result.json")
    GT_FILE = os.path.join(BASE_DIR, "dataset", "sa_lzk_data", "cna_valid_ground_truth.json")
    run_evaluation(PRED_FILE, GT_FILE, is_test_mode=True)