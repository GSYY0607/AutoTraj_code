# 设置环境变量，让 Ray/vLLM/PyTorch 用 /hy-tmp 作为临时目录
export TMPDIR=/hy-tmp/tmp
export TEMP=/hy-tmp/tmp
export TMP=/hy-tmp/tmp

# 创建目录
mkdir -p $TMPDIR


export PYTHONPATH=/to/your/path/Tool-Star/Tool_Star_RL/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU



bash /to/your/path/Tool-Star/Tool_Star_RL/scripts/train/train.sh \
    --train_batch_size 12 \
    --ppo_mini_batch_size 6 \
    --rollout_n 4 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path /to/your/path/model \
    --project_name your name \
    --experiment_name your name \
    --nnodes 1 \
    --n_gpus_per_node 4 \
    --save_freq 60 \
    --test_freq 60 \
    --total_epochs 1 \
    --wandb_api_key your key \
    --save_path /to/your/path/model \
    --train_files /to/your/path/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet \
    --test_files /to/your/path/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet
