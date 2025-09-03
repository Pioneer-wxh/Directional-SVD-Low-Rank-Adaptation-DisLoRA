#!/bin/bash

# ==========================================================================================
# GLUE Benchmark Experiment Runner Script
# ==========================================================================================
#
# 这个脚本会自动遍历指定的GLUE任务和参数高效微调（PEFT）方法，
# 并为每个组合运行 GLUEBenchmark.py 脚本。
#
# 如何使用:
# 1. 将此脚本保存为 `run_experiments.sh`，并与 `GLUEBenchmark.py` 放在同一目录下。
# 2. 确保 `MyLoRA.py` 文件也在同一目录中。
# 3. 在终端中给予此脚本执行权限: `chmod +x run_experiments.sh`
# 4. 运行脚本: `./run_experiments.sh`
#
# 您可以根据需要修改 TASKS 和 METHODS 数组来运行不同的实验组合。
# ==========================================================================================

# -- 配置实验参数 --

# 定义要运行的 GLUE 任务列表 (用空格分隔)
# 可选项: "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
export HF_ENDPOINT=https://hf-mirror.com
TASKS=("cola" )

# 定义要测试的 PEFT 方法列表 (用空格分隔)
# 可选项: "MyLoRA", "LoRA", "DoRA", "PiSSA"
METHODS=("MyLoRA" )

# 定义通用的超参数
LEARNING_RATE=1e-4
LORA_RANK=16
LORA_ALPHA=24
BATCH_SIZE=8

ORTHO_LAMBDA=0.1 # 仅对 MyLoRA 生效

# -- 脚本主循环 --

# 记录开始时间
start_time=$(date +%s)
echo "🚀 开始运行 GLUE 基准测试实验..."
echo "========================================"

# 遍历任务列表
for task in "${TASKS[@]}"
do
    # 遍历方法列表
    for method in "${METHODS[@]}"
    do
        # 打印当前正在运行的实验配置
        echo ""
        echo "------------------------------------------------"
        echo "📊 正在运行任务: [${task}], 方法: [${method}]"
        echo "------------------------------------------------"

        # 构建并执行 Python 脚本命令
        # 注意：--ortho_lambda 参数仅在 method 为 MyLoRA 时有实际意义，但为了简化脚本，我们统一传入。
        # Python 脚本内部的逻辑会决定是否使用它。
        python GLUEBenchmark.py \
            --task "$task" \
            --method "$method" \
            --lr "$LEARNING_RATE" \
            --r "$LORA_RANK" \
            --lora_alpha "$LORA_ALPHA" \
            --batch_size "$BATCH_SIZE" \
            --ortho_lambda "$ORTHO_LAMBDA"

        # 检查上一个命令是否成功执行
        if [ $? -ne 0 ]; then
            echo "❌ 实验失败: 任务 [${task}], 方法 [${method}]"
            # 如果希望在某个实验失败后立即停止整个脚本，可以取消下面这行的注释
            # exit 1
        else
            echo "✅ 实验完成: 任务 [${task}], 方法 [${method}]"
        fi
    done
done

# 记录结束时间并计算总耗时
end_time=$(date +%s)
total_seconds=$((end_time - start_time))
echo "========================================"
echo "🎉 所有实验已全部完成！"
echo "⏱️  总耗时: ${total_seconds} 秒。"
echo "========================================"