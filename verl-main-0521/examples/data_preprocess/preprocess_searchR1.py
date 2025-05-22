"""
Preprocess the Search-R1 dataset (from HuggingFace) to parquet format
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import datasets

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
                    # "execute_kwargs": {},
                    # "calc_reward_kwargs": {},
                    # "release_kwargs": {},
                },
            },
        }

        return {
            "id_searchR1": id_searchR1,
            "question": question,
            "golden_answers": golden_answers ,
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
    parser.add_argument("--local_dir", default="~/data/searchR1")
    parser.add_argument("--hdfs_dir", default=None)
    

    args = parser.parse_args()

    hf_dataset = "PeterJinGo/nq_hotpotqa_train"
    dataset = datasets.load_dataset(hf_dataset, split="train")
    

    # Map and transform
    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset.get("test")
    if test_dataset:
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Output paths
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    if test_dataset:
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Saved to {local_dir}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS {args.hdfs_dir}")
