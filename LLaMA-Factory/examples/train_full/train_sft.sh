# module load cuda/12.1.1
cd /to/your/path/LLaMA-Factory

# 可选：设置 WandB 项目名和运行名
export WANDB_PROJECT="your WANDB_PROJECT"
export WANDB_RUN_NAME="your RUN_NAME"

CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /to/your/path/LLaMA-Factory/examples/train_full/qwen_sft_autotraj.yaml
