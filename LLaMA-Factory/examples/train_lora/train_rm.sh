#!/bin/bash
# reward_model_train_wandb.sh
# 用于启动 Qwen LoRA 奖励模型训练，并支持 WandB 上报

# 进入 LLaMA-Factory 主目录
cd /to/your/path/LLaMA-Factory

# 设置可见显卡，例如用全部4张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 可选：设置 WandB 项目名和运行名
export WANDB_PROJECT="your project name"
export WANDB_RUN_NAME="your run name"

# 启动训练
llamafactory-cli train /to/your/path/LLaMA-Factory/examples/train_lora/qwen_lora_reward.yaml
