

datasets=('AQuA')


for dataset_name in "${datasets[@]}"
do
  echo "======================================="
  echo "Running evaluation for dataset: $dataset_name"
  echo "======================================="

  CUDA_VISIBLE_DEVICES=$2 python math_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter DisLoRA \
    --dataset "$dataset_name" \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 5 \
    --lora_weights $1 | tee -a $1/${dataset_name}.txt

  echo "Finished evaluation for dataset: $dataset_name"
  echo ""
done

echo "All evaluations finished."