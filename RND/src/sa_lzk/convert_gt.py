# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict
from pypinyin import pinyin, Style

def clean_pinyin(s):
    """去掉声调并转为纯小写"""
    # pypinyin 返回的是列表嵌套，需要取 [0][0]
    return "".join([item[0] for item in pinyin(s, style=Style.NORMAL)]).lower()

from pypinyin import pinyin, Style

def convert_to_snake_pinyin(name_cn):
    """
    将中文姓名转换为 姓_名_名 格式 (全小写)
    例如: '吴东清' -> 'wu_dong_qing', '李松' -> 'li_song'
    """
    if not name_cn:
        return ""
    # 获取拼音列表: [['wu'], ['dong'], ['qing']]
    pinyin_list = pinyin(name_cn, style=Style.NORMAL)
    # 转换为下划线连接的字符串
    return "_".join([item[0].lower() for item in pinyin_list])

def run_conversion(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}")
        return

    print(f"正在读取原始数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        unass_data = json.load(f)

    # 结构: {name_key: {auth_id: [paper_id, ...]}}
    gt_dict = defaultdict(lambda: defaultdict(list))

    for item in unass_data:
        name_cn = item.get('name', '')
        auth_id = item.get('gh', '')    # 新数据中的作者ID字段
        paper_id = item.get('wos', '')  # 新数据中的论文ID字段

        if not name_cn or not auth_id or not paper_id:
            continue

        # 转换为拼音格式: li_hongbing
        name_key = convert_to_name_key(name_cn)
        
        # 填充到字典中，实现同名同ID论文聚合
        gt_dict[name_key][auth_id].append(paper_id)

    # 写入保存至目标路径
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt_dict, f, indent=4, ensure_ascii=False)
    
    print(f"转换完成！")
    print(f"成功生成符合格式的 GT 文件: {output_path}")
    print(f"处理样本总数: {len(unass_data)}")
    print(f"生成姓名簇数量: {len(gt_dict)}")

if __name__ == "__main__":
    # 1. 获取当前脚本的绝对路径: RND/src/sa_lzk/convert_gt.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 向上跳转两级到达 RND 根目录
    # 第一级: RND/src
    # 第二级: RND
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 3. 定义输入输出地址
    # 输入: RND/dataset/sa_lzk_data/unass.json
    INPUT_PATH = os.path.join(project_root, "dataset", "sa_lzk_data", "unass.json")
    # 输出: RND/dataset/sa_lzk_data/cna_valid_ground_truth.json
    OUTPUT_PATH = os.path.join(project_root, "dataset", "sa_lzk_data", "cna_valid_ground_truth.json")

    run_conversion(INPUT_PATH, OUTPUT_PATH)