export WANDB_API_KEY=7a5831d26740f1e82f08b0878d05950f8cbf727e

# 定义时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}

# 1. 生成一次时间戳，并存储起来
current_timestamp=$(now)

# 2. 构建日志文件名和实验名后缀
log_file="logs/searchR1-${current_timestamp}.log"
experiment_name_with_timestamp="qwen2.5-3b_instruct_rm-searchR1-like-sgl-multiturn-${current_timestamp}"
# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs



nohup bash examples/sglang_multiturn/searchR1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
    trainer.experiment_name=${experiment_name_with_timestamp} \
    > "${log_file}" 2>&1 &


tail -f "${log_file}"