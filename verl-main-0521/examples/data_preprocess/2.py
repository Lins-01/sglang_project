import argparse
import os
import datasets
import pandas as pd
from functools import partial # 用于向 apply 的函数传递额外参数

# 假设 hdfs_io_1.py 与此脚本在同一 Python 包路径下，或者 PYTHONPATH 配置正确
from .hdfs_io_1 import copy, makedirs

# 系统与用户前缀内容
system_content = "You are a helpful and harmless assistant."
user_content_prefix = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "


def process_row_custom(row, split_name):
    """
    Processes a single row of a Pandas DataFrame.
    The output structure is strictly aligned with the first script's make_map_fn.
    'row' is a Pandas Series representing a row from the DataFrame.
    'idx' (index) is obtained from row.name.
    'split_name' is passed to indicate 'train' or 'test'.
    """
    idx = row.name # Get index from the DataFrame row

    # Extract data from the input row (Pandas Series)
    # These field names should match the columns from the HuggingFace dataset
    id_searchR1_original = row.get("id", "")
    question_original = row.get("question", "")
    golden_answers_original = row.get("golden_answers", [])
    data_source_original = row.get("data_source", "")
    ability_original = row.get("ability", "")
    reward_model_original = row.get("reward_model", {})
    metadata_original = row.get("metadata", None)

    # Construct 'prompt' field
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content_prefix + question_original}
    ]

    # Construct 'data_source' field (tagged)
    data_source_tagged = "searchR1_" + str(data_source_original)

    # Construct 'extra_info' field
    extra_info = {
        "split": split_name,
        "index": idx,
        "answer": golden_answers_original, # Using golden_answers from input
        "question": question_original,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "search": {
                "create_kwargs": {"ground_truth": golden_answers_original}, # Using golden_answers
            },
        },
    }

    # Return a Pandas Series with the desired output structure
    # The keys of this Series will become the column names in the new DataFrame
    return pd.Series({
        "id_searchR1": id_searchR1_original,
        "question": question_original,
        "golden_answers": golden_answers_original,
        "data_source": data_source_tagged,
        "prompt": prompt,
        "ability": ability_original,
        "reward_model": reward_model_original,
        "extra_info": extra_info,
        "metadata": metadata_original
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Search-R1 dataset from HuggingFace to Parquet format using Pandas.")
    parser.add_argument("--hf_dataset_name", default="PeterJinGo/nq_hotpotqa_train",
                        help="Name of the dataset on HuggingFace Hub.")
    parser.add_argument("--local_dir", default="~/data/searchR1_processed_pandas",
                        help="Local directory to save the Parquet files.")
    parser.add_argument("--hdfs_dir", default=None,
                        help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    # --- TRAIN DATASET ---
    processed_train_df = None
    print(f"Loading train split from {args.hf_dataset_name}...")
    try:
        train_dataset_raw = datasets.load_dataset(args.hf_dataset_name, split="train")
        if train_dataset_raw:
            print("Converting train dataset to Pandas DataFrame...")
            train_df_raw = train_dataset_raw.to_pandas()
            print(f"Processing train DataFrame (Size: {len(train_df_raw)})...")
            # Use functools.partial to pass the split_name to process_row_custom
            # or a lambda function: lambda row: process_row_custom(row, split_name="train")
            process_row_train = partial(process_row_custom, split_name="train")
            processed_train_df = train_df_raw.apply(process_row_train, axis=1)
            print("Train dataset processing complete.")
    except Exception as e:
        print(f"Error loading or processing train split for {args.hf_dataset_name}: {e}")

    # --- TEST DATASET ---
    processed_test_df = None
    print(f"Attempting to load test split from {args.hf_dataset_name}...")
    try:
        test_dataset_raw = datasets.load_dataset(args.hf_dataset_name, split="test")
        if test_dataset_raw:
            print("Converting test dataset to Pandas DataFrame...")
            test_df_raw = test_dataset_raw.to_pandas()
            print(f"Processing test DataFrame (Size: {len(test_df_raw)})...")
            process_row_test = partial(process_row_custom, split_name="test")
            processed_test_df = test_df_raw.apply(process_row_test, axis=1)
            print("Test dataset processing complete.")
    except Exception as e:
        print(f"Could not load or process test split for {args.hf_dataset_name}: {e}")
        print("Proceeding without test dataset.")

    # Output paths
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if processed_train_df is not None and not processed_train_df.empty:
        train_output_path = os.path.join(local_dir, "train.parquet")
        print(f"Saving processed train DataFrame to {train_output_path}...")
        processed_train_df.to_parquet(train_output_path, index=False)
        print(f"Train dataset saved to {train_output_path}")

    if processed_test_df is not None and not processed_test_df.empty:
        test_output_path = os.path.join(local_dir, "test.parquet")
        print(f"Saving processed test DataFrame to {test_output_path}...")
        processed_test_df.to_parquet(test_output_path, index=False)
        print(f"Test dataset saved to {test_output_path}")
    elif processed_train_df is not None: # Only print if train was processed but test was not
        print("No test dataset was processed or available to save.")


    if (processed_train_df is None or processed_train_df.empty) and \
       (processed_test_df is None or processed_test_df.empty):
        print("No data was processed. Exiting.")
    else:
        print(f"All processed files saved locally to {local_dir}")

        if args.hdfs_dir:
            print(f"Copying files to HDFS directory: {args.hdfs_dir}...")
            try:
                makedirs(args.hdfs_dir) # Ensure HDFS directory exists
                copy(src=local_dir, dst=args.hdfs_dir) # Copy contents of local_dir to hdfs_dir
                print(f"Successfully copied files to HDFS: {args.hdfs_dir}")
            except Exception as e:
                print(f"Error copying files to HDFS: {e}")