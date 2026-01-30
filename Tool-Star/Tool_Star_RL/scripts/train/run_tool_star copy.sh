



export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU


bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path /hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-SFT \
    --project_name AutoTraj \
    --experiment_name AutoTraj_GRPO_7B \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --wandb_api_key d2069356704ee1b8ce4f1f5f101c2d43ab9fd70d \
    --save_path /hy-tmp/AutoTraj/models/Qwen2.5-7B-Instruct-GRPO \
    --train_files /hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet \
    --test_files /hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet
