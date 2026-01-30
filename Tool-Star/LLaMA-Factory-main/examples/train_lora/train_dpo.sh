# module load cuda/12.1.1
cd ./LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /LLaMA-Factory-main/examples/train_lora/qwen_lora_dpo_2.yaml