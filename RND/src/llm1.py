# -*- coding: utf-8 -*-
import dspy
import os
import asyncio
from transformers import AutoTokenizer 

TOKENIZER_DIR = r"D:\download\deepseek_v3_tokenizer\deepseek_v3_tokenizer"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    tokenizer.model_max_length = 100000
except Exception as e:
    print(f"Tokenizer 加载失败: {e}")
    tokenizer = None

def get_token_count(text):
    """利用官方分词器计算精确长度"""
    if tokenizer and text:
        try:
            tokens = tokenizer.encode(
                str(text), 
                add_special_tokens=False, 
                truncation=False
            )
            return len(tokens)
        except Exception:
            # 如果极端情况报错，回退到字符估算（学术场景 1 token ≈ 3-4 字符）
            return len(str(text)) // 4 
    return 0
class DisambiguationSignature(dspy.Signature):
    prompt = dspy.InputField(desc="完整任务描述")
    
    confidence_level = dspy.OutputField(desc="置信度评分 (1-6)")
    best_id = dspy.OutputField(desc="匹配成功的 ID,或 'new_author'")
    reasoning = dspy.OutputField(desc="不超过200字")

class Disambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 Predict 提高 V3 的响应速度，如需分析则用 ChainOfThought
        self.predictor = dspy.Predict(DisambiguationSignature)
    
    def __call__(self, prompt):
        return self.predictor(prompt=prompt)

async def ask_deepseek_async(task_id, paper_info, candidate_profiles, target_name, current_index=0, total_count=0):
    """
    异步封装层：利用 dspy.asyncify 实现并发调用
    """
    authors_list = paper_info.get('authors', [])
    num_candidates = len(candidate_profiles)
    all_authors = [a.get('name', '') for a in paper_info.get('authors', [])]
    co_authors = [name for name in all_authors if name != target_name]
    target_org = "N/A"
    for auth in authors_list:
        if auth.get('name') == target_name:
            target_org = auth.get('org', 'N/A')
            break
    
    paper_text = f"论文标题: {paper_info.get('title', 'N/A')}\n"
    paper_text += f"待消歧作者机构: {target_org}\n"
    paper_text += f"合作者 (Exclude Target): {', '.join(co_authors)}\n"
    paper_text += f"发表时间: {paper_info.get('year', 'N/A')} | 发表渠道: {paper_info.get('venue', 'N/A')}\n"
    keywords = paper_info.get("keywords", [])
    if keywords:
        paper_text += f"论文关键词: {', '.join(keywords)}"
    paper_text += f"摘要: {paper_info.get('abstract', 'N/A')[:200]}"

    profiles_text = "\n".join([f"【ID: {k}】\n{v}" for k, v in candidate_profiles.items()])
    in_tokens = get_token_count(paper_text + profiles_text)

    model = Disambiguator()
    async_model = dspy.asyncify(model)
    
    try:
        prompt = f"""
[任务目标]
判定论文作者是否属于候选人池中的某个学者。

禁止无依据推测，必须基于明确证据。

[判定优先级]
1. 合作者（最重要）
2. 机构 = 领域
3. 时间
4. 期刊

【论文信息】
{paper_text}

【候选人画像】
{profiles_text}

【输出要求】
confidence_level: 1-6
best_id: ID 或 new_author
reasoning: 不超过200字

严格按格式输出，不要额外内容。
"""

        prediction = await async_model(prompt=prompt)
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