# -*- coding: utf-8 -*-
import dspy
import re
import json
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
    基于合作者、关键词和机构快速评价候选人与论文的匹配潜力，为第二阶段决策进行初筛。数据特点：候选人大部分来自同一机构，机构区分度较低，仅作为弱辅助特征。
    
    [置信度判定要求]
    - Level 5:
        - 存在合作者重合
        OR
        - 标题高度相似（核心词一致）且关键词高度契合
    - Level 4:
        - 关键词（领域）深度契合
        OR
        - 标题语义高度相似
    - Level 3:
        - 关键词高度契合，但无标题或合作者证据
    - Level 2:
        - 关键词部分重合（泛领域相关）
    - Level 1:
        - 领域、标题、合作者完全无关
    """
    paper_info = dspy.InputField(desc="待处理论文的标题、核心合作者、关键词、机构")
    candidate_briefs = dspy.InputField(desc="候选人精简列表（ID, 关键词, 机构, 核心合作者）")
    results = dspy.OutputField(desc="必须严格按格式输出: ID:Level_X, ID:Level_X... (例如 1001:Level_5, 1002:Level_3)")

class L2DeepAnalysis(dspy.Signature):
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


    def forward(self, paper_text, candidate_profiles_dict,gt_id=None,current_index=0, total_count=0, mode="strict"):

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

        l1_hit = 0
        is_nil_gt = (gt_id is None)

        if is_nil_gt:
            l1_hit = 1
        else:
            if gt_id in top_ids:
                l1_hit = 1
            else:
                l1_hit = 0

        if not top_ids:
            if mode == "strict":
                print(f"[{current_index}/{total_count}] [第一层粗筛结束] 未发现匹配候选人，直接终止。")
                return dspy.Prediction(
                    confidence_level="1",
                    best_id="new_author",
                    reasoning="第一阶段(L1)粗筛未发现任何在领域、机构或合作者方面具有关联的候选人。",
                    stage_stats={
                        "l1_hit": l1_hit,
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
            "l1_hit": l1_hit,
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



async def ask_deepseek_two_stage_async(task_id, paper_info, candidate_profiles, target_name, gt_id=None,current_index=0, total_count=0):
    """
    异步封装层
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
        prediction = await async_model(paper_text=paper_text, candidate_profiles_dict=candidate_profiles,gt_id=gt_id,current_index=current_index, total_count=total_count)
        
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
            out_tokens,
            s["l1_hit"],
            (gt_id is None)
        )
    except Exception as e:
        print(f" 任务 {task_id} 两层调用异常: {e}")
        return (task_id, None, str(e), 0, 0, 0, 0, 0, 0, (gt_id is None))