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

    # 2. 建立全量真值索引：paper_id -> author_id
    paper_to_author_map = {}
    for name, authors in ground_truth.items():
        for auth_id, papers in authors.items():
            for pid in papers:
                paper_to_author_map[pid] = auth_id

    # 3. 整理预测结果：author_id -> {paper_id}
    author_pred_papers = defaultdict(set)
    all_task_papers = set() 
    for pred_auth_id, task_ids in predictions.items():
        for task_id in task_ids:
            pid = task_id.split('-')[0]
            author_pred_papers[pred_auth_id].add(pid)
            all_task_papers.add(pid)

    # 4. 核心：根据这 100 篇论文确定 M 个作者
    # 逻辑：只要论文不在 paper_to_author_map 里，它就是 NIL 这一组的
    active_true_authors = set()
    author_true_papers = defaultdict(set)

    for pid in all_task_papers:
        correct_auth_id = paper_to_author_map.get(pid)
        if correct_auth_id is not None:
            # 该论文属于已知作者
            active_true_authors.add(correct_auth_id)
            author_true_papers[correct_auth_id].add(pid)
        else:
            # 重要：GT 中找不到，说明正确答案是 NIL
            active_true_authors.add("new_author")
            author_true_papers["new_author"].add(pid)

    # 5. 指标统计 (严格对齐公式)
    # M = 此时 active_true_authors 只包含这 100 题涉及的真实作者（含NIL）
    all_author_ids = active_true_authors 
    TUP = len(all_task_papers) if is_test_mode else 13914
    
    WeightedPrecision = 0
    WeightedRecall = 0

    for auth_id in all_author_ids:
        true_set = author_true_papers[auth_id]
        pred_set = author_pred_papers[auth_id] # 你的模型预测给该 ID 的论文
        
        # --- 变量对应公式 ---
        CPA = len(true_set & pred_set)    # CorrectlyPredictedToTheAuthor
        TPA = len(pred_set)               # TotalPredictedToTheAuthor
        UPA = len(true_set)               # UnassignedPaperOfTheAuthor
        
        Precision_i = CPA / TPA if TPA > 0 else 0
        Recall_i = CPA / UPA if UPA > 0 else 0
        Weight_i = UPA / TUP
        
        # 加权求和: sum(Metric_i * weight_i)
        WeightedPrecision += Precision_i * Weight_i
        WeightedRecall += Recall_i * Weight_i

        
        display_id = "NIL (new_author)" if auth_id == "new_author" else auth_id
        print(f"{str(display_id):<25} | {CPA:<5} | {TPA:<5} | {UPA:<5} | {Precision_i:<8.1%} | {Recall_i:<8.1%}")

    # 6. 计算最终 WeightedF1
    if (WeightedPrecision + WeightedRecall) > 0:
        WeightedF1 = 2 * (WeightedPrecision * WeightedRecall) / (WeightedPrecision + WeightedRecall)
    else:
        WeightedF1 = 0

    # 7. 保留 NIL 监控指标
    nil_true = author_true_papers.get("new_author", set())
    nil_pred = author_pred_papers.get("new_author", set())
    nil_recall = len(nil_true & nil_pred) / len(nil_true) if len(nil_true) > 0 else 0
    
    old_author_papers_count = len(all_task_papers - nil_true)
    nil_false_positives = len(nil_pred - nil_true) # 误报：老作者被你预测成了新作者
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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRED_FILE = os.path.join(BASE_DIR, "output", "result.json")
    GT_FILE = os.path.join(BASE_DIR, "dataset", "valid", "cna_valid_ground_truth.json")
    
    run_evaluation(PRED_FILE, GT_FILE, is_test_mode=True)