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
    prompt = dspy.InputField(desc="完整任务描述")
    results = dspy.OutputField(desc="ID:Level_X 格式")


class L2DeepAnalysis(dspy.Signature):
    prompt = dspy.InputField(desc="完整分析任务")
    
    confidence_level = dspy.OutputField(desc="1-6")
    best_id = dspy.OutputField(desc="ID 或 new_author")
    reasoning = dspy.OutputField(desc="不超过200字")

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

        l1_prompt = f"""
[任务目标]
判定论文作者是否属于候选人。

允许弱匹配（提高召回），禁止无依据猜测。

[规则]
- Level 5: 合作者重合
- Level 4: 关键词深度契合 + 机构一致
- Level 3: 关键词契合
- Level 2: 部分相关
- Level 1: 无关

【论文信息】
{paper_text}

【候选人】
{l1_cands_text}

【输出格式】
1001:Level_5, 1002:Level_3

不要解释。
"""

        l1_res = self.l1_filter(prompt=l1_prompt)
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

        l2_prompt = f"""
[任务]
执行作者消歧最终决策

[优先级]
合作者 > 机构 = 领域 > 时间

【阶段信息】
{stage_context}

【论文信息】
{paper_text}

【候选人画像】
{l2_profiles_text}

【输出格式】
confidence_level: 1-6
best_id: ID 或 new_author
reasoning: 不超过200字

严格按格式输出
"""

        res = self.l2_analyzer(prompt=l2_prompt)
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
        out_tokens = get_token_count(prediction.best_id + prediction.reasoning)

        return (
            task_id,                
            final_id,               
            prediction.reasoning,   
            prediction.stage_stats["l1_cands"],          
            prediction.stage_stats["l2_cands"],         
            prediction.stage_stats["two_stage_total_input_tokens"], 
            original_in_tokens,     
            out_tokens,
            prediction.stage_stats["l1_hit"],
            (gt_id is None)
        )
    except Exception as e:
        print(f" 任务 {task_id} 两层调用异常: {e}")
        return (task_id, None, str(e), 0, 0, 0, 0, 0, 0, (gt_id is None))