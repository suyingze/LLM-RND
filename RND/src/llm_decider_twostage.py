# -*- coding: utf-8 -*-
import dspy
import re
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
    if tokenizer and text:
        try:
            tokens = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
            return len(tokens)
        except Exception:
            return len(str(text)) // 4 
    return 0


class L1LightweightFilter(dspy.Signature):
    """
    [任务目标]
    判定待处理论文的作者是否指向候选人池中的特定学者。
    本任务为身份判定任务，禁止基于缺失信息进行合理性补全或推测，所有正向证据必须来源于明确给出的事实。若某一判断依赖于“可能是”“通常是”“看起来像”，则该判断视为无效证据。

    [第一阶段：分级粗筛]
    基于合作者、关键词和机构快速评价候选人与论文的匹配潜力，为第二阶段决策进行初筛。
    
    [置信度判定要求]
    - Level 5: 存在合作者重合；
    - Level 4: 关键词（领域）深度契合，且机构一致；
    - Level 3: 关键词（领域）高度契合，但机构缺失或不一致；
    - Level 2: 关键词仅部分重合（泛领域相关），无其他证据；
    - Level 1: 领域、机构、合作者完全无关。
    """
    paper_info = dspy.InputField(desc="待处理论文的标题、核心合作者、关键词、机构")
    candidate_briefs = dspy.InputField(desc="候选人精简列表（ID, 关键词, 机构, 核心合作者）")
    results = dspy.OutputField(desc="必须严格按格式输出: ID:Level_X, ID:Level_X... (例如 1001:Level_5, 1002:Level_3)")

class L2DeepAnalysis(dspy.Signature):
    """
    [第二阶段：全特征深度决策]
    [判定准则与权重优先级]
    1. 核心指纹 (Highest) - 合作者网络: 
       - 合作者重合是匹配的铁证。如果存在多个候选者与论文合作者重合，则优先考虑合作者出现的频率。频率高者优先
       -【严格限定】仅当合作者姓名在字符串层面完全一致（忽略大小写与标点）时，才能认定为“合作者重合”。
       - 姓名顺序互换（如 “Tao Hong” vs “Hong Tao”）、中英文顺序差异、或拼写相似，均不得单独作为合作者重合的证据，除非存在额外的明确事实（如相同机构 + 同一论文）。
    2. 机构一致性 (High, 与领域连续性同权重): 
       - 机构匹配具有强补强作用，优先级等同于领域（关键词），高于单纯的时间线吻合。
       - 但仍需注意：若论文机构为空，禁止仅因此判定为 'new_author'。也不得推理补全，必须基于明确的事实。
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
    stage_context = dspy.InputField(
        desc="第一阶段筛选状态说明，由系统给定，不需要模型判断"
    )
    paper_info = dspy.InputField(desc="含标题、合作者、机构、年份、关键词及摘要的信息")
    candidate_profiles = dspy.InputField(desc="候选人画像池(包含合作者合作频率以及关键词频率等细粒度特征)")
    
    confidence_level = dspy.OutputField(desc="置信度评分 (1-6)")
    best_id = dspy.OutputField(desc="匹配成功的 ID,或 'new_author'")
    reasoning = dspy.OutputField(desc="简要说明判断逻辑与依据,不超过200字")


class TwoStageDisambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.l1_filter = dspy.Predict(L1LightweightFilter)
        self.l2_analyzer = dspy.Predict(L2DeepAnalysis)

    def _parse_and_truncate(self, results_str):
        # 修改正则以匹配 "Level_X" 格式
        pattern = r"([^,:\s]+)\s*:\s*Level_?(\d)"
        matches = re.findall(pattern, results_str)
        if not matches: return []
    
        scored_ids = sorted([(m[0], int(m[1])) for m in matches], key=lambda x: x[1], reverse=True)
        final_ids = [item[0] for item in scored_ids if item[1] >= 2]
        return final_ids


    def forward(self, paper_text, candidate_profiles_dict,current_index=0, total_count=0, mode="strict"):

        l1_cands_list = []
        for k, v in candidate_profiles_dict.items():
            orgs_section = v.split("- orgs:")[1].split("- keywords:")[0].strip() if "- orgs:" in v else "N/A"
            kws_line = v.split("- keywords: ")[1].split("\n")[0].strip() if "- keywords: " in v else "N/A"
            cols_line = v.split("- collaborators: ")[1].split("\n")[0].strip() if "- collaborators: " in v else "N/A"
            l1_cands_list.append(
                f"ID:{k}\nOrgs:\n{orgs_section}\nKeywords: {kws_line}\nCollaborators: {cols_line}"
            )

        l1_cands_text = "\n\n".join(l1_cands_list)
        l1_in_tokens = get_token_count(paper_text + l1_cands_text)

        print(f"[{current_index}/{total_count}] [第一层粗筛结束] 初始候选人: {len(candidate_profiles_dict)} | Tokens: {l1_in_tokens}")

        l1_res = self.l1_filter(paper_info=paper_text, candidate_briefs=l1_cands_text)
        top_ids = self._parse_and_truncate(l1_res.results)

        if not top_ids:
            if mode == "strict":
                print(f"[{current_index}/{total_count}] [第一层粗筛结束] 未发现匹配候选人，直接终止。")
                return dspy.Prediction(
                    confidence_level="1",
                    best_id="new_author",
                    reasoning="第一阶段(L1)粗筛未发现任何在领域、机构或合作者方面具有关联的候选人。",
                    stage_stats={
                        "two_stage_total_input_tokens": l1_in_tokens,
                        "l1_cands": len(candidate_profiles_dict),
                        "l1_tokens": l1_in_tokens,
                        "l2_cands": 0,
                        "l2_tokens": 0
                    }
                )

            elif mode == "fallback":
                stage_context = (
                    "第一阶段轻量筛选未筛选出任何入围候选人，"
                    "未发现具有明确弱匹配信号的对象，"
                    "候选人集合保持完整。"
                    "请基于全部候选人进行严格评估，"
                    "若无充分证据必须返回 'new_author'，"
                    "禁止生成候选列表之外的任何作者 ID。"
                )
                filtered_profiles = candidate_profiles_dict

        else:
            stage_context = (
                "第一阶段轻量筛选已筛选出入围候选人，"
                "以下候选人为高潜力对象，"
                "请优先基于这些候选进行评估。" 
                "若无充分证据必须返回 'new_author'，"
                "禁止生成候选列表之外的任何作者 ID。"
            )
            filtered_profiles = {
                k: v for k, v in candidate_profiles_dict.items()
                if str(k) in top_ids
            }

        l2_profiles_text = "\n".join(filtered_profiles.values())
        l2_in_tokens = get_token_count(paper_text + l2_profiles_text)

        print(f"[{current_index}/{total_count}] [第二层深度分析开始] 输入候选人: {len(filtered_profiles)} | Tokens: {l2_in_tokens}")

        stats = {
            "two_stage_total_input_tokens": l1_in_tokens + l2_in_tokens,
            "l1_cands": len(candidate_profiles_dict),
            "l1_tokens": l1_in_tokens,
            "l2_cands": len(filtered_profiles),
            "l2_tokens": l2_in_tokens,
            "mode": mode,
            "l1_empty": int(not top_ids)
        }

        res = self.l2_analyzer(
            stage_context=stage_context,
            paper_info=paper_text,
            candidate_profiles=l2_profiles_text
        )
        res.stage_stats = stats
        return res


async def ask_deepseek_two_stage_async(task_id, paper_info, candidate_profiles, target_name, current_index=0, total_count=0):
    """
    异步封装层：与单层版保持参数一致
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
    paper_text += f"发表时间: {paper_info.get('year', 'N/A')} | 渠道: {paper_info.get('venue', 'N/A')}\n"
    paper_text += f"摘要: {paper_info.get('abstract', 'N/A')[:200]}"
    keywords = paper_info.get("keywords", [])
    if keywords:
        paper_text += f"论文关键词: {', '.join(keywords)}"

    profiles_text = "\n".join([f"【ID: {k}】\n{v}" for k, v in candidate_profiles.items()])
    original_in_tokens = get_token_count(paper_text + profiles_text)
    
    model = TwoStageDisambiguator()
    async_model = dspy.asyncify(model)
    
    try:
        # 执行两层推理
        prediction = await async_model(paper_text=paper_text, candidate_profiles_dict=candidate_profiles,current_index=current_index, total_count=total_count)
        
        out_tokens = get_token_count(prediction.best_id + prediction.reasoning)
        res_id = prediction.best_id.strip().replace("'", "").replace('"', "")
        
        # 结果标准化
        if res_id.upper() in ["NIL", "NONE", "NEW_AUTHOR", "NULL"]:
            final_id = "new_author"
        else:
            final_id = res_id
        s = prediction.stage_stats    
        return (
            task_id,                
            final_id,               
            prediction.reasoning,   
            s["l1_cands"],          
            s["l2_cands"],         
            s["two_stage_total_input_tokens"], 
            original_in_tokens,     
            out_tokens
        )
    except Exception as e:
        print(f" 任务 {task_id} 两层调用异常: {e}")
        raise e