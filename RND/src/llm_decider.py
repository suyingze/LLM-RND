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

    """
    [任务目标]
    判定待处理论文的作者是否指向候选人池中的特定学者。
    本任务为身份判定任务，禁止基于缺失信息进行合理性补全或推测，所有正向证据必须来源于明确给出的事实。若某一判断依赖于“可能是”“通常是”“看起来像”，则该判断视为无效证据。

    [判定准则与权重优先级]
    1. 核心指纹 (Highest) - 合作者网络: 
       - 合作者重合是匹配的铁证。如果存在多个候选者与论文合作者重合，则优先考虑合作者出现的频率。频率高者优先
       -【严格限定】仅当合作者姓名在字符串层面完全一致（忽略大小写与标点）时，才能认定为“合作者重合”。
       - 姓名顺序互换（如 “Tao Hong” vs “Hong Tao”）、中英文顺序差异、或拼写相似，均不得单独作为合作者重合的证据，除非存在额外的明确事实（如相同机构 + 同一论文）。
    2. 机构一致性 (High, 与领域连续性同权重): 
       - 机构匹配具有强补强作用，优先级等同于领域（关键词），高于单纯的时间线吻合。
       - 但仍需注意：若论文待消歧作者机构为空，禁止仅因此判定为 'new_author'。也不得推理补全，必须基于明确的事实。
    3. 领域连续性 (High, 与机构一致性同权重): 
       - 核心领域或关键词的深度契合是重要依据。若存在多个候选者研究领域契合，则关键词出现频率高者优先。若关键词为空，利用摘要进行领域推断。
    4. 时间线与活跃度 (Medium): 
       - 检查发表年份是否在候选人职业活跃期内。
    5.发表渠道(Low) :
       - 发表在候选人常见的会议/期刊上可以作为加分项，但不可单独作为肯定或否定作者身份的依据。

    [置信度判定要求]
    - Level 6 (极高): 合作者匹配且机构和领域一致。
    - Level 5 (高): 合作者匹配，且机构高度匹配；或合作者匹配且领域高度契合。
    - Level 4 (较高): 存在合作者匹配，但机构不同且不存在明确的时间或地点冲突；或在存在合作者的前提下，领域存在一定差异但未跨越学科边界。
    - Level 3 (中): 无合作者但领域相近，机构大致匹配。
    - Level 2 (较低): 无合作者，领域相对较远，机构不匹配但无直接冲突，通常不足以支持身份确认，允许返回 'NIL'。
    - Level 1 (排除): 领域、机构完全无关或存在地点/时间硬冲突。

    [输出要求]
    若置信度 >= 3,必须返回最可能的 ID。只有在 Level 1以及Level 2时才允许返回 'NIL'。
    """
    paper_info = dspy.InputField(desc="含标题、合作者、机构、年份、关键词及摘要的信息")
    candidate_profiles = dspy.InputField(desc="候选人画像池（包含合作者合作频率以及关键词频率等细粒度特征）")
    
    confidence_level = dspy.OutputField(desc="置信度评分 (1-6)")
    best_id = dspy.OutputField(desc="匹配成功的 ID,或 'new_author'")
    reasoning = dspy.OutputField(desc="简要说明判断逻辑与依据,不超过200字")

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