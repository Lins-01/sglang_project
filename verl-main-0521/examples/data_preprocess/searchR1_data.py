"""
Preprocess the Search-R1 dataset (from HuggingFace) to parquet format
"""

import argparse
import os
import re # re 模块未在你的原始代码中使用，如果确实不用可以考虑移除
import numpy as np # numpy 未在你的原始代码中使用，如果确实不用可以考虑移除
import pandas as pd # pandas 未在你的原始代码中使用，如果确实不用可以考虑移除
import datasets

# 假设 hdfs_io_1.py 与此脚本在同一目录下或在 Python 路径中
# 如果 hdfs_io_1.py 在同目录，运行时仍需 python -m your_module.your_script
from .hdfs_io_1 import copy, makedirs


# 系统与用户前缀内容
system_content = "You are a helpful and harmless assistant."
user_content_prefix = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "


def make_map_fn(split):
    def process_fn(example, idx):
        id_searchR1 = example.get("id", "")
        question = example.get("question", "")
        golden_answers = example.get("golden_answers", [])
        data_source_searchR1 = example.get("data_source", "")
        ability = example.get("ability", "")
        reward_model = example.get("reward_model", {})
        extra_info_searchR1 = example.get("extra_info", {})
        metadata = example.get("metadata", None)

        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content_prefix + question}
        ]

        data_source_tagged = "searchR1_" + data_source_searchR1

        extra_info = {
            "split": split,
            "index": idx,
            "answer": golden_answers,
            "question": question,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {"ground_truth": golden_answers},
                },
            },
        }

        return {
            "id_searchR1": id_searchR1,
            "question": question,
            "golden_answers": golden_answers,
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": ability,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": metadata
        }
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/searchR1_processed", help="Directory to save processed files") # 修改了默认输出目录名以区分
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy processed files to")
    
    # --- 开始修改 ---
    # 1. 添加新的命令行参数来指定本地 Parquet 文件路径
    parser.add_argument("--local_train_file", type=str, help="Path to your local train.parquet file")
    parser.add_argument("--local_test_file", type=str, default=None, help="Path to your local test.parquet file (optional)")
    # --- 结束修改 ---

    args = parser.parse_args()

    # --- 开始修改 ---
    # 2. 构建 data_files 字典
    args.local_train_file = r"E:\Document\CodeSpace\Study\DeepL\RL_Project\verl-main-0521\examples\data_preprocess\train.parquet"
    # args.local_test_file = r"E:\Document\CodeSpace\Study\DeepL\RL_Project\verl-main-0521\examples\data_preprocess\test.parquet"
    data_files = {}
    if args.local_train_file:
        if not os.path.exists(args.local_train_file):
            raise FileNotFoundError(f"Train file not found: {args.local_train_file}")
        data_files['train'] = args.local_train_file
        print(f"Using local train file: {args.local_train_file}")
    
    if args.local_test_file:
        if not os.path.exists(args.local_test_file):
            raise FileNotFoundError(f"Test file not found: {args.local_test_file}")
        data_files['test'] = args.local_test_file
        print(f"Using local test file: {args.local_test_file}")

    if not data_files:
        raise ValueError("No local data files provided. Please specify at least --local_train_file.")

    # 3. 从本地 Parquet 文件加载数据集
    # `load_dataset` 使用 `parquet` 类型，并传入 `data_files` 字典
    # 这会返回一个 DatasetDict，其中键是 'train', 'test' 等
    dataset_dict = datasets.load_dataset('parquet', data_files=data_files)
    # --- 结束修改 ---
    
    # Map and transform train data
    if "train" in dataset_dict:
        train_dataset = dataset_dict["train"].map(function=make_map_fn("train"), with_indices=True)
        
        # Output paths for train data
        local_dir = os.path.expanduser(args.local_dir)
        os.makedirs(local_dir, exist_ok=True)
        
        train_output_path = os.path.join(local_dir, "processed_train.parquet") # 给处理后的文件一个新名字
        train_dataset.to_parquet(train_output_path)
        print(f"Saved processed train data to {train_output_path}")
    else:
        print("No train data loaded or processed.")

    # Map and transform test data (if provided and you want to process it)
    if "test" in dataset_dict:
        test_dataset = dataset_dict["test"].map(function=make_map_fn("test"), with_indices=True)
        
        local_dir = os.path.expanduser(args.local_dir) # 确保目录存在
        os.makedirs(local_dir, exist_ok=True)

        test_output_path = os.path.join(local_dir, "processed_test.parquet") # 给处理后的文件一个新名字
        test_dataset.to_parquet(test_output_path)
        print(f"Saved processed test data to {test_output_path}")

    # HDFS copy (if args.hdfs_dir is provided)
    if args.hdfs_dir:
        if not os.path.exists(local_dir) or not os.listdir(local_dir):
             print(f"Local directory {local_dir} is empty or does not exist. Skipping HDFS copy.")
        else:
            makedirs(args.hdfs_dir) # 确保 HDFS 目标目录存在 (你的 hdfs_io_1.makedirs 实现)
            copy(src=local_dir, dst=args.hdfs_dir) # 你的 hdfs_io_1.copy 实现
            print(f"Copied processed files from {local_dir} to HDFS {args.hdfs_dir}")