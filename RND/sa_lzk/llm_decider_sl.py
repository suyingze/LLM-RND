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
    [第二阶段：全特征深度决策]

    [判定准则与权重优先级]

    1. 核心指纹 (Highest)
       - 合作者网络（铁证）
       - 标题语义一致性（新增：与合作者同级的重要证据）
       
       规则：
       - 合作者完全字符串匹配才算
       - 标题若核心词高度一致，可作为强匹配信号

    2. 领域连续性 (High)
       - 关键词重合程度
       - 候选人历史关键词频率匹配
       - 若关键词缺失，可用摘要辅助判断

    3. 机构一致性 (Low, 降权)
       - 当前为校内数据，机构区分度低
       - 仅用于辅助，不可主导判断
       - 禁止因机构一致直接提升置信度

    4. 时间线与活跃度 (Medium)
       - 是否在候选人活跃期内

    5. 发表渠道 (Low)
       - 仅作为弱辅助

    --------------------------------
    [置信度判定要求]

    - Level 6 (极高):
        合作者匹配 + 标题高度一致 + 领域一致

    - Level 5 (高):
        合作者匹配 + 领域高度契合
        OR 标题高度一致 + 关键词高度一致

    - Level 4 (较高):
        存在合作者匹配但领域略有差异
        OR 标题语义相似 + 关键词匹配

    - Level 3 (中):
        无合作者，但标题或领域高度一致

    - Level 2 (较低):
        无合作者，领域较远，仅部分关键词匹配

    - Level 1 (排除):
        标题、领域、合作者均无关

    --------------------------------
    [输出要求]

    若置信度 >= 3, 必须返回最可能的 ID  
    Level 1 / Level 2 才允许返回 'new_author'

    --------------------------------
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