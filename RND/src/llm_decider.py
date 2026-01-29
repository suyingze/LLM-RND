# -*- coding: utf-8 -*-
import dspy
import os
import asyncio
from transformers import AutoTokenizer 

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
    """
    姓名消歧判定准则：
     核心目标：判断待处理论文的作者是否为候选人池中的某一位。
    请按以下权重优先级进行判定：
    1. 【首要权重】合作者重合度：这是判定同一人的最强证据。
    2. 【次要权重】研究领域的延续性：关注核心技术领域的关联。
    3. 【修正权重】机构一致性：机构仅作为辅助参考。若论文机构为空，请完全忽略此项差异，不得据此判定为 new_author。
    4. 【否定项】只有在发现明确的物理冲突（如地点时间完全重叠且不可兼得）时，才考虑判定为 new_author。

    """
    paper_info = dspy.InputField(desc="待处理论文的标题、机构等")
    candidate_profiles = dspy.InputField(desc="候选人背景画像池，注意：候选人画像中括号内的数字代表该特征出现的频次。频次越高，代表该特征（如合作者或研究主题）对该候选人的代表性越强。")
    best_id = dspy.OutputField(desc="匹配成功的 ID 或 'new_author'")
    reasoning = dspy.OutputField(desc="判定依据（简要） ")

class Disambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 Predict 提高 V3 的响应速度，如需分析则用 ChainOfThought
        self.predictor = dspy.Predict(DisambiguationSignature)
    
    def __call__(self, paper_info, candidate_profiles):
        return self.predictor(paper_info=paper_info, candidate_profiles=candidate_profiles)

async def ask_deepseek_async(task_id, paper_info, candidate_profiles, current_index=0, total_count=0):
    """
    异步封装层：利用 dspy.asyncify 实现并发调用
    """
    # --- 监控 A: 基础数据准备 ---
    num_candidates = len(candidate_profiles)
    paper_text = str(paper_info)
    profiles_text = "\n".join([f"【ID: {k}】\n{v}" for k, v in candidate_profiles.items()])
    
    # 计算输入 Token
    in_tokens = get_token_count(paper_text + profiles_text)
    
    # 打印时使用“已提交”字样，因为异步模式下多个任务会同时显示在这里
    print(f"[{current_index}/{total_count}] 🚀 任务提交: {task_id} | 候选人: {num_candidates} | Tokens: {in_tokens}")

    # --- 监控 B: 异步调用 DSPy ---
    model = Disambiguator()
    
    # 使用 dspy.asyncify 将同步模块转为异步
    # 这允许我们在 await 时释放事件循环，让其他任务也能启动
    async_model = dspy.asyncify(model)
    
    try:
        # 异步等待结果返回
        prediction = await async_model(paper_info=paper_text, candidate_profiles=profiles_text)
        
        # --- 监控 C: 结果处理 ---
        out_tokens = get_token_count(prediction.best_id + prediction.reasoning)
        
        # 清洗结果
        res_id = prediction.best_id.strip().replace("'", "").replace('"', "")
        
        # 统一规范化：如果是 NIL、None 或 NEW_AUTHOR，统一返回 None 供逻辑判断
        # 注意：这里增加对 NEW_AUTHOR 的识别
        if res_id.upper() in ["NIL", "NONE", "NEW_AUTHOR"]:
            final_id = None
        else:
            final_id = res_id
            
        return final_id, prediction.reasoning, num_candidates, in_tokens, out_tokens

    except Exception as e:
        print(f"❌ 任务 {task_id} API 调用异常: {e}")
        # 向上抛出异常，让 main.py 的 try-except 捕获
        raise e