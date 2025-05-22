import argparse
import os
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError # 用于捕获文件未找到的错误
import tempfile
from functools import partial

# 假设 hdfs_io_1.py 与此脚本在同一 Python 包路径下，或者 PYTHONPATH 配置正确
# 根据你提供的代码，我们保留原样
# 如果 hdfs_io_1.py 在同目录下且不是包的一部分，你可能需要调整导入方式
# 例如 from hdfs_io_1 import copy, makedirs
from .hdfs_io_1 import copy, makedirs


# 系统与用户前缀内容 (与你提供的脚本一致)
system_content = "You are a helpful and harmless assistant."
user_content_prefix = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "


def process_single_row(row, current_split_name, row_index):
    """
    处理单行数据，逻辑参考你提供的“不报错的本地版本代码”，但适配从HuggingFace原始文件加载的场景。
    """
    question = row.get('question', '')

    # 1. 处理prompt
    user_content = user_content_prefix + question
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # 2. 从reward_model获取ground_truth，如果失败则尝试golden_answers
    # PeterJinGo/nq_hotpotqa_train 的 schema 显示 reward_model 是一个 struct
    # 我们也期望它有 golden_answers 列作为备选
    ground_truth = None
    reward_model_data = row.get('reward_model') # pd.read_parquet 可能会将其读为 dict 或 Series
    
    if isinstance(reward_model_data, dict) and 'ground_truth' in reward_model_data:
        ground_truth = reward_model_data.get('ground_truth')
    
    if ground_truth is None: # 如果无法从reward_model获取，或reward_model非预期格式
        ground_truth = row.get('golden_answers', []) # 假设存在 golden_answers 列

    # 3. 处理data_source
    data_source_original = row.get('data_source', '') # 假设存在 data_source 列
    data_source_tagged = 'searchR1_' + str(data_source_original)

    # 4. 构建tools_kwargs (结构参考你“不报错的本地版本代码”)
    tools_kwargs = {
        'search': {
            'create_kwargs': {
                'ground_truth': ground_truth,
                'question': question,
                'data_source': data_source_tagged
            }
        }
    }

    # 5. 构建完整的extra_info (结构参考你“不报错的本地版本代码”，但index和split动态生成)
    extra_info = {
        'index': row_index, # 使用 DataFrame 的行索引
        'need_tools_kwargs': True,
        'question': question,
        'split': current_split_name, # 使用当前处理的 split 名称
        'tools_kwargs': tools_kwargs
    }

    # 6. 按照“不报错的本地版本代码”的格式顺序返回
    return pd.Series({
        'data_source': data_source_tagged,
        'prompt': prompt,
        'ability': row.get('ability'), # 假设存在 ability 列
        'reward_model': reward_model_data, # 保留原始的 reward_model 数据
        'extra_info': extra_info,
        'metadata': row.get('metadata') # 假设存在 metadata 列
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument("--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID.")
    parser.add_argument("--local_dir", default="~/data/searchR1_processed_direct", help="Local directory to save the processed Parquet files.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")
    
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_any_file = False

    # 使用临时目录下载HuggingFace的Parquet文件
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train", "test"]:
            parquet_filename = f"{split}.parquet"
            print(f"\nProcessing {split} split...")
            
            try:
                # 1. 从HuggingFace下载Parquet文件
                print(f"Downloading {parquet_filename} from {args.hf_repo_id}...")
                local_parquet_filepath = hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=parquet_filename,
                    repo_type="dataset",
                    local_dir=tmp_download_dir,
                    local_dir_use_symlinks=False # 建议在Windows上设为False避免权限问题
                )
                print(f"Downloaded to {local_parquet_filepath}")

                # 2. 使用Pandas读取Parquet文件
                df_raw = pd.read_parquet(local_parquet_filepath)
                print(f"Loaded {len(df_raw)} rows from {parquet_filename} into DataFrame.")

                if df_raw.empty:
                    print(f"{split} split is empty. Skipping processing.")
                    continue

                # 3. 应用处理函数
                # 使用 lambda 和 row.name 来传递行索引
                print(f"Applying processing function to {split} data...")
                df_processed = df_raw.apply(
                    lambda row: process_single_row(row, current_split_name=split, row_index=row.name),
                    axis=1
                )
                
                if df_processed.empty:
                    print(f"No data after processing for {split} split. Skipping save.")
                    continue

                # 4. 保存处理后的DataFrame
                output_file_path = os.path.join(local_save_dir, f"{split}.parquet") # 与原脚本输出文件名一致
                df_processed.to_parquet(output_file_path, index=False)
                print(f"Successfully processed and saved {split} data to {output_file_path}")
                processed_any_file = True

            except EntryNotFoundError:
                print(f"{parquet_filename} not found in repository {args.hf_repo_id}. Skipping {split} split.")
            except Exception as e:
                print(f"An error occurred while processing {split} split: {e}")
                import traceback
                traceback.print_exc()

    if not processed_any_file:
        print("No data was processed or saved.")
    else:
        print(f"\nAll processed files saved locally to {local_save_dir}")
        # 5. HDFS 复制 (如果需要)
        if args.hdfs_dir:
            if os.path.exists(local_save_dir) and os.listdir(local_save_dir):
                print(f"Copying processed files from {local_save_dir} to HDFS directory: {args.hdfs_dir}...")
                try:
                    makedirs(args.hdfs_dir) # 确保HDFS目录存在
                    copy(src=local_save_dir, dst=args.hdfs_dir) # 复制 local_save_dir 的内容到 hdfs_dir
                    print(f"Successfully copied files to HDFS: {args.hdfs_dir}")
                except Exception as e:
                    print(f"Error copying files to HDFS: {e}")
            else:
                print(f"Local save directory {local_save_dir} is empty or does not exist. Skipping HDFS copy.")