# -*- coding: utf-8 -*-
import os

# --- 1. 实验全局配置 ---
# 可选模式: "title", "title_keywords", "title_venue", "title_abstract"
# 每次跑不同的特征对比实验时，只需修改这一行
CURRENT_FEATURE_MODE = "title_keywords" 

# --- 2. 路径管理 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 建议向量缓存根目录
VECTOR_CACHE_ROOT = os.path.join(BASE_DIR, "..", "output", "vector_cache")

# 文件夹简写映射（让目录名短一点，方便在终端查看）
MODE_DIR_MAP = {
    "title": "T",
    "title_keywords": "T_K",
    "title_venue": "T_V",
    "title_abstract": "T_A",
    "keywords_venue": "K_V",          
    "title_keywords_venue": "T_K_V"   
}

def get_vector_cache_path():
    """根据当前特征模式，自动返回对应的子目录路径"""
    sub_dir = MODE_DIR_MAP.get(CURRENT_FEATURE_MODE, "T_K")
    target_path = os.path.join(VECTOR_CACHE_ROOT, sub_dir)
    # 自动创建不存在的子目录
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    return target_path

# --- 3. 核心逻辑：特征提取函数 ---
def build_feature_text(pub_detail, mode=None):
    """
    统一预处理和推理脚本的文本拼接逻辑。
    如果 mode 为 None，则使用全局定义的 CURRENT_FEATURE_MODE。
    """
    active_mode = mode if mode else CURRENT_FEATURE_MODE
    
    title = pub_detail.get('title', '').strip()
    # 兼容关键词列表和字符串格式
    kws = pub_detail.get('keywords', [])
    keywords = " ".join(kws) if isinstance(kws, list) else str(kws)
    
    venue = pub_detail.get('venue', '').strip()
    abstract = pub_detail.get('abstract', '').strip()

    if active_mode == "title":
        return title
    elif active_mode == "title_keywords":
        return f"{title} {keywords}".strip()
    elif active_mode == "title_venue":
        return f"{title} {venue}".strip()
    elif active_mode == "title_abstract":
        return f"{title} {abstract}".strip()
    elif active_mode == "keywords_venue":
        return f"{keywords} {venue}".strip()
    elif active_mode == "title_keywords_venue":
        return f"{title} {keywords} {venue}".strip()
    
    
    return title  # 默认降级方案