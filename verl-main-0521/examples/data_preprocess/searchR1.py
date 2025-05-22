import pandas as pd
import json
import numpy as np

# 设置系统消息内容
system_content = "You are a helpful and harmless assistant."
user_content_prefix = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <function> query </function> and it will return the top searched results between <|quad_start|> and <|quad_end|>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "

# 文件路径
input_parquet = r"E:\Document\CodeSpace\Study\DeepL\RL_Project\verl-main-0521\examples\data_preprocess\test.parquet"
output_parquet = r"E:\Document\CodeSpace\Study\DeepL\RL_Project\verl-main-0521\examples\data_preprocess\test_2.parquet"
jsonl_path =r"E:\Document\CodeSpace\Study\DeepL\RL_Project\verl-main-0521\examples\data_preprocess\test.jsonl"

# 读取Parquet文件
df = pd.read_parquet(input_parquet)
# 只处理0:500条
# df = df.iloc[:500]

# 处理各行数据
def process_row(row):
    # 1. 处理prompt - 将numpy数组转为列表并添加系统消息
    user_content = user_content_prefix + row['question']
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # 2. 从reward_model获取ground_truth
    try:
        if isinstance(row['reward_model'], dict) and 'ground_truth' in row['reward_model']:
            ground_truth = row['reward_model']['ground_truth']
        else:
            ground_truth = None
    except (KeyError, TypeError):
        ground_truth = None

    # 3. 构建extra_info
    question = row.get('question', '')
    index = row['extra_info'].get('index') if isinstance(row['extra_info'], dict) else None
    split = row['extra_info'].get('split') if isinstance(row['extra_info'], dict) else 'train'

    # 4. 处理data_source
    data_source = 'searchR1_' + str(row['data_source'])

    # 构建tools_kwargs
    tools_kwargs = {
        'search': {
            'create_kwargs': {
                'ground_truth': ground_truth,
                'question': question,
                'data_source': data_source
            }
        }
    }

    # 构建完整的extra_info
    extra_info = {
        'index': index,
        'need_tools_kwargs': True,
        'question': question,
        'split': split,
        'tools_kwargs': tools_kwargs
    }

    # 按照目标格式顺序返回
    return pd.Series({
        'data_source': data_source,
        'prompt': prompt,
        'ability': row['ability'],
        'reward_model': row['reward_model'],
        'extra_info': extra_info,
        'metadata': row.get('metadata')
    })

# 应用转换
df = df.apply(process_row, axis=1)

# 保存到Parquet
df.to_parquet(output_parquet, index=False)

# JSON序列化器处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# 随机保存200条记录到JSONL
records_200 = df.sample(n=200, random_state=42).to_dict(orient='records')
with open(jsonl_path, 'w', encoding='utf-8') as f:
    for rec in records_200:
        f.write(json.dumps(rec, ensure_ascii=False, cls=NumpyEncoder) + '\n')

print(f"保存对齐数据到 {output_parquet}")
print(f"随机保存200条记录到JSONL文件 {jsonl_path}")
