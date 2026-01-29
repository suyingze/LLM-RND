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
    [任务目标]
    判定待处理论文的作者是否指向候选人池中的特定学者。

    [判定准则与权重优先级]
    1. 核心指纹 (Highest) - 合作者网络: 
       - 合作者重合是匹配的铁证。
    2. 领域连续性 (High): 
       - 核心领域或关键词的深度契合是关键依据。
    3. 机构一致性 (Medium-High): 
       - 机构匹配具有强补强作用，优先级高于单纯的时间线吻合。
       - 但仍需注意：若论文机构为空，禁止仅因此判定为 'new_author'。
    4. 时间线与活跃度 (Medium): 
       - 检查发表年份是否在候选人职业活跃期内。

    [置信度判定要求]
    - Level 5 (极高): 合作者匹配且机构/领域一致。
    - Level 4 (高): 无合作者，但机构完全匹配且领域高度契合。
    - Level 3 (中): 领域完全吻合，虽然机构不同，但无硬性物理冲突。或者领域不符合但是有合作者。
    - Level 2 (低): 领域相近，但缺乏任何合作者或机构支撑。
    - Level 1 (排除): 领域完全无关或存在地点/时间硬冲突。

    [输出要求]
    若置信度 >= 3，必须返回最可能的 ID。只有在 Level 1 时才允许返回 'NIL'。
    """
    paper_info = dspy.InputField(desc="含标题、合作者、机构、年份及摘要的信息")
    candidate_profiles = dspy.InputField(desc="候选人画像池")
    
    confidence_level = dspy.OutputField(desc="置信度评分 (1-5)")
    best_id = dspy.OutputField(desc="匹配成功的 ID，或 'new_author'")
    reasoning = dspy.OutputField(desc="简要说明判断逻辑与依据")

class Disambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 Predict 提高 V3 的响应速度，如需分析则用 ChainOfThought
        self.predictor = dspy.Predict(DisambiguationSignature)
    
    def __call__(self, paper_info, candidate_profiles):
        return self.predictor(paper_info=paper_info, candidate_profiles=candidate_profiles)

async def ask_deepseek_async(task_id, paper_info, candidate_profiles, target_name, current_index=0, total_count=0):
    """
    异步封装层：利用 dspy.asyncify 实现并发调用
    """
    num_candidates = len(candidate_profiles)
    all_authors = [a.get('name', '') for a in paper_info.get('authors', [])]
    
    co_authors = [name for name in all_authors if name != target_name]
    
    
    paper_text = f"论文标题: {paper_info.get('title', 'N/A')}\n"
    paper_text += f"合作者 (Exclude Target): {', '.join(co_authors)}\n"
    paper_text += f"发表时间: {paper_info.get('year', 'N/A')} | 发表渠道: {paper_info.get('venue', 'N/A')}\n"
    
    abstract = paper_info.get('abstract', '')
    paper_text += f"摘要简述: {abstract[:150]}..." if abstract else "摘要: N/A"

    profiles_text = "\n".join([f"【ID: {k}】\n{v}" for k, v in candidate_profiles.items()])
    
    in_tokens = get_token_count(paper_text + profiles_text)
    
    print(f"[{current_index}/{total_count}]  任务提交: {task_id} | 候选人: {num_candidates} | Tokens: {in_tokens}")

    model = Disambiguator()
    
    async_model = dspy.asyncify(model)
    
    try:
        prediction = await async_model(paper_info=paper_text, candidate_profiles=profiles_text)
        
        out_tokens = get_token_count(prediction.best_id + prediction.reasoning)
        
        res_id = prediction.best_id.strip().replace("'", "").replace('"', "")
        
        if res_id.upper() in ["NIL", "NONE", "NEW_AUTHOR"]:
            final_id = None
        else:
            final_id = res_id
            
        return final_id, prediction.reasoning, num_candidates, in_tokens, out_tokens

    except Exception as e:
        print(f" 任务 {task_id} API 调用异常: {e}")

        raise e