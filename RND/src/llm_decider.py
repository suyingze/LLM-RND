# -*- coding: utf-8 -*-
import dspy
import os
from transformers import AutoTokenizer 

# 1. 初始化 Tokenizer 
TOKENIZER_DIR = r"D:\download\deepseek_v3_tokenizer\deepseek_v3_tokenizer"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
except Exception as e:
    print(f"Tokenizer 加载失败: {e}")
    tokenizer = None

def get_token_count(text):
    """利用官方分词器计算精确长度"""
    if tokenizer and text:
        return len(tokenizer.encode(str(text)))
    return 0

class DisambiguationSignature(dspy.Signature):
    """姓名消歧判定：根据论文信息匹配候选人 ID 或返回 new_author。"""
    paper_info = dspy.InputField(desc="待处理论文的标题、机构等")
    candidate_profiles = dspy.InputField(desc="候选人背景画像池，注意：候选人画像中括号内的数字代表该特征出现的频次。频次越高，代表该特征（如合作者或研究主题）对该候选人的代表性越强。")
    best_id = dspy.OutputField(desc="匹配成功的 ID 或 'new_author'")
    reasoning = dspy.OutputField(desc="判定依据（简要）")

class Disambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 Predict 提高 V3 的响应速度，如需分析则用 ChainOfThought
        self.predictor = dspy.Predict(DisambiguationSignature)
    
    def __call__(self, paper_info, candidate_profiles):
        return self.predictor(paper_info=paper_info, candidate_profiles=candidate_profiles)

def ask_deepseek(task_id, paper_info, candidate_profiles,current_index=0, total_count=0):
    """
    封装层：负责数据监控、计数与 DSPy 调用
    """
    # --- 监控 A: 候选人个数 ---
    num_candidates = len(candidate_profiles)
    
    # --- 监控 B: 输入 Token 统计 ---
    paper_text = str(paper_info)
    profiles_text = "\n".join([f"【ID: {k}】\n{v}" for k, v in candidate_profiles.items()])
    in_tokens = get_token_count(paper_text + profiles_text)
    
    print(f"\n[{current_index}] 正在处理任务:{task_id} | 候选人个数: {num_candidates} | 输入 Tokens: {in_tokens}")

    # 调用 DSPy
    model = Disambiguator()
    prediction = model(paper_info=paper_text, candidate_profiles=profiles_text)
    
    # --- 监控 C: 输出 Token 统计 ---
    out_tokens = get_token_count(prediction.best_id + prediction.reasoning)
    
    # 清洗结果
    res_id = prediction.best_id.strip().replace("'", "").replace('"', "")
    final_id = None if res_id.upper() in ["NIL", "NONE"] else res_id
    
    return final_id, prediction.reasoning, num_candidates, in_tokens, out_tokens