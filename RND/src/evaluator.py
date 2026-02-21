# -*- coding: utf-8 -*-
import json
import os

def run_evaluation(pred_path, gt_path, is_test_mode=True): #is_test_mode: 是否为测试模式。True时以当前预测数做分母，False时以全量任务数做分母
    
    # 1. 加载数据
    if not os.path.exists(pred_path):
        print(f"错误：找不到预测文件 {pred_path}")
        return
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)  # {auth_id: [task_id, ...]}
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f) # {name: {auth_id: [paper_id, ...]}}

    # 2. 建立反向索引：将标准答案展平为 paper_id -> author_id
    # 如果某篇论文不在这个索引里，说明它在官方定义中属于 NIL (新作者) 
    paper_to_author = {}
    for name, authors in ground_truth.items():
        for auth_id, papers in authors.items():
            for pid in papers:
                paper_to_author[pid] = auth_id

    # 3. 统计指标
    tp = 0  # 正确预测数 (True Positives)
    total_preds = 0
    nil_correct = 0 # 正确识别新作者的数量
    nil_total_in_sample = 0 # 样本中实际为新作者的总量
    nil_false = 0  # 误报数
    old_total_in_sample = 0    # 真实老作者总数
    
    print("\n" + "="*60)
    print(f"{'任务ID':<15} | {'你的预测':<15} | {'标准答案':<15} | {'状态'}")
    print("-" * 60)
    
    for pred_auth_id, task_ids in predictions.items():
        for task_id in task_ids:
            total_preds += 1
            paper_id = task_id.split('-')[0]
            correct_auth_id = paper_to_author.get(paper_id) # 若没找到则为 None (NIL) 
            
            is_correct = False
            # 1. 先判定总体正确性
            is_correct = (pred_auth_id == correct_auth_id) or \
                (pred_auth_id == "new_author" and correct_auth_id is None)

            # 2. 专门针对 NIL (新作者) 的统计逻辑（采用 Recall 维度：真实的新作者里你找对了多少）
            if correct_auth_id is None:
                nil_total_in_sample += 1  # 只要事实是新作者，分母就加 1
                if pred_auth_id == "new_author":
                    nil_correct += 1      # 只有你预测对了，分子才加 1

            if correct_auth_id is not None:  # 事实是：老作者（已知专家）
                old_total_in_sample += 1
                if pred_auth_id == "new_author": # 错误：模型却说是新作者
                    nil_false += 1

            # 3. 统计 TP
            if is_correct:
                tp += 1
                status = " 正确"
                display_gt = correct_auth_id if correct_auth_id else "NIL"
            else:
                status = " 错误"
                display_gt = correct_auth_id if correct_auth_id else "*NIL*"
            
            print(f"{task_id:<15} | {pred_auth_id:<15} | {str(display_gt):<15} | {status}")

    # 4. 计算分数 
    # 在测试阶段，Precision 和 Recall 分母均使用 total_preds，反映当前样本的准确性
    precision = tp / total_preds if total_preds > 0 else 0
    
    
    recall_denominator = total_preds if is_test_mode else 13914 
    recall = tp / recall_denominator if recall_denominator > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("="*60)
    print(f" 测试阶段评估报告 (Mode: {'Test' if is_test_mode else 'Full'}):")
    print(f"  > 已完成任务总数: {total_preds}")
    print(f"  > 预测正确总数: {tp}")
    print(f"  > 局部准确率 (Precision): {precision:.2%}")
    print(f"  > 局部召回率 (Recall): {recall:.2%}")
    print(f"  > 综合 F1 分数: {f1:.4f}")
    if nil_total_in_sample > 0:
        print(f"  > 新作者(NIL)召回率: {nil_correct/nil_total_in_sample:.2%}")
    fpr = nil_false / old_total_in_sample if old_total_in_sample > 0 else 0
    print(f" > 新作者误报率 : {fpr:.2%}")
    print("="*60)


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRED_FILE = os.path.join(BASE_DIR, "output", "result.json")
    GT_FILE = os.path.join(BASE_DIR, "dataset", "valid", "cna_valid_ground_truth.json")
    
    run_evaluation(PRED_FILE, GT_FILE, is_test_mode=True)
    