

export HF_ENDPOINT=https://hf-mirror.com
TASKS=("cola" )


METHODS=("DisLoRA" )


LEARNING_RATE=1e-4
LORA_RANK=16
LORA_ALPHA=24
BATCH_SIZE=8

ORTHO_LAMBDA=0.1 


start_time=$(date +%s)
echo "🚀 Starting GLUE benchmark experiments..."
echo "========================================"


for task in "${TASKS[@]}"
do

    for method in "${METHODS[@]}"
    do

        echo ""
        echo "------------------------------------------------"
        echo "📊 Running task: [${task}], method: [${method}]"
        echo "------------------------------------------------"


        python GLUEBenchmark.py \
            --task "$task" \
            --method "$method" \
            --lr "$LEARNING_RATE" \
            --r "$LORA_RANK" \
            --lora_alpha "$LORA_ALPHA" \
            --batch_size "$BATCH_SIZE" \
            --ortho_lambda "$ORTHO_LAMBDA"


        if [ $? -ne 0 ]; then
            echo "❌ Experiment failed: task [${task}], method [${method}]"

        else
            echo "✅ Experiment completed: task [${task}], method [${method}]"
        fi
    done
done


end_time=$(date +%s)
total_seconds=$((end_time - start_time))
echo "========================================"
echo "🎉 All experiments completed!"
echo "⏱️  Total time: ${total_seconds} seconds."
echo "========================================"