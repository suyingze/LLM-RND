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
    """利用官方分词器计算长度。"""
    if tokenizer and text:
        try:
            tokens = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
            return len(tokens)
        except Exception:
            return len(str(text)) // 4
    return 0


class SimpleConcatSignature(dspy.Signature):
    prompt = dspy.InputField(desc="简单拼接后的任务文本")
    best_id = dspy.OutputField(desc="候选人ID或new_author")
    reasoning = dspy.OutputField(desc="不超过200字")


class SimpleConcatFallbackSignature(dspy.Signature):
    prompt = dspy.InputField(desc="简单拼接后的任务文本")
    output = dspy.OutputField(desc="自由文本输出，包含best_id和reasoning")


class SimpleConcatDisambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleConcatSignature)

    def __call__(self, prompt):
        return self.predictor(prompt=prompt)


class SimpleConcatFallbackDisambiguator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleConcatFallbackSignature)

    def __call__(self, prompt):
        return self.predictor(prompt=prompt)


def _extract_fields(prediction_obj):
    """兼容结构化输出失败时的自由文本解析。"""
    best_id = ""
    reasoning = ""

    if hasattr(prediction_obj, "best_id"):
        best_id = (getattr(prediction_obj, "best_id", "") or "").strip()
    if hasattr(prediction_obj, "reasoning"):
        reasoning = (getattr(prediction_obj, "reasoning", "") or "").strip()
    if best_id and reasoning:
        return best_id, reasoning

    raw_text = ""
    if hasattr(prediction_obj, "output"):
        raw_text = str(getattr(prediction_obj, "output", "") or "")
    else:
        raw_text = str(prediction_obj)

    id_match = re.search(r"best_id\s*[:：]\s*([^\n\r]+)", raw_text, flags=re.IGNORECASE)
    reason_match = re.search(r"reasoning\s*[:：]\s*([^\n\r]+)", raw_text, flags=re.IGNORECASE)

    if id_match:
        best_id = id_match.group(1).strip()
    if reason_match:
        reasoning = reason_match.group(1).strip()

    if not best_id:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if lines:
            best_id = lines[0]
        if len(lines) > 1 and not reasoning:
            reasoning = lines[1]

    if not reasoning:
        reasoning = raw_text[:200]

    return best_id, reasoning


async def ask_deepseek_async(task_id, paper_info, candidate_profiles, target_name, current_index=0, total_count=0):
    """
    简单拼接基线:
    - 不提供证据优先级
    - 不提供置信度分级规则
    - 仅要求输出 best_id 和 reasoning
    """
    authors_list = paper_info.get("authors", [])
    num_candidates = len(candidate_profiles)
    all_authors = [a.get("name", "") for a in authors_list]
    co_authors = [name for name in all_authors if name != target_name]

    target_org = "N/A"
    for auth in authors_list:
        if auth.get("name") == target_name:
            target_org = auth.get("org", "N/A")
            break

    paper_text = f"论文标题: {paper_info.get('title', 'N/A')}\n"
    paper_text += f"待消歧作者机构: {target_org}\n"
    paper_text += f"合作者 (Exclude Target): {', '.join(co_authors)}\n"
    paper_text += f"发表时间: {paper_info.get('year', 'N/A')} | 发表渠道: {paper_info.get('venue', 'N/A')}\n"
    keywords = paper_info.get("keywords", [])
    if keywords:
        paper_text += f"论文关键词: {', '.join(keywords)}\n"
    paper_text += f"摘要: {paper_info.get('abstract', 'N/A')[:200]}"

    profiles_text = "\n".join([f"【ID: {k}】\n{v}" for k, v in candidate_profiles.items()])
    in_tokens = get_token_count(paper_text + profiles_text)

    model = SimpleConcatDisambiguator()
    async_model = dspy.asyncify(model)
    fallback_model = SimpleConcatFallbackDisambiguator()
    fallback_async_model = dspy.asyncify(fallback_model)

    prompt = f"""请根据下面信息，判断论文作者最可能对应哪个候选人ID。
如果都不匹配，输出 new_author。

【论文信息】
{paper_text}

【候选人画像】
{profiles_text}

请严格输出两个字段：
best_id: <候选人ID或new_author>
reasoning: <简要理由，不超过200字>
"""

    try:
        print(f"[{current_index}/{total_count}] [SimpleConcat] 开始请求，候选人: {num_candidates}")
        try:
            prediction = await asyncio.wait_for(async_model(prompt=prompt), timeout=120)
        except Exception:
            prediction = await asyncio.wait_for(fallback_async_model(prompt=prompt), timeout=120)

        best_id, reasoning = _extract_fields(prediction)
        out_tokens = get_token_count((best_id or "") + (reasoning or ""))
        res_id = (best_id or "").strip().replace("'", "").replace('"', "")

        if res_id.upper() in ["NIL", "NONE", "NEW_AUTHOR", ""]:
            final_id = None
        else:
            final_id = res_id

        print(f"[{current_index}/{total_count}] [SimpleConcat] 请求完成，输出ID: {final_id if final_id else 'NIL'}")
        return final_id, reasoning, num_candidates, in_tokens, out_tokens
    except asyncio.TimeoutError:
        print(f"[{current_index}/{total_count}] [SimpleConcat] 请求超时(120s): {task_id}")
        return None, "Timeout in simple_concat", num_candidates, in_tokens, 0
    except Exception as e:
        print(f"任务 {task_id} API 调用异常: {e}")
        raise e
