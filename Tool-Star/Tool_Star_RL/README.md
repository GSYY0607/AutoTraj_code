
## 🔥 Self-Critic RL Stage

In this step, we will load the cold-start data for GRPO training. We reference the [ReCall](https://github.com/Agent-RL/ReCall) and [VERL](https://github.com/volcengine/verl) frameworks for RL training.

### 1. Environment Setup

First, please set up the VERL environment. After that, install our environment:

```bash
# Create conda environment
conda create -n tool_star python=3.10
conda activate tool_star

# Install requirements
cd tool_star
pip install -r requirements.txt
```

### 2. Vanilla RL Training

Our training framework is based on [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). The training scripts can be found under `scripts/train`. First, you need to complete the information in `scripts/train/run_tool_star.sh`:

```bash
export PYTHONPATH=/src/verl:$PYTHONPATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

bash scripts/train/train.sh \
    --train_batch_size 128 \
    --ppo_mini_batch_size 16 \
    --rollout_n 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path {your_actor_model_path} \
    --project_name {your_project_name} \
    --experiment_name {your_experiment_name} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --wandb_api_key {your_wandb_api_key} \
    --save_path {your_save_path} \
    --train_files {path_to_train_file}/grpo_mix_train_shuffle.parquet \
    --test_files {path_to_test_file}/grpo_mix_test.parquet
```

Since the rollout process involves Bing web search calls, please configure the `deep_search_snippet()` function in `/src/verl/verl/workers/rollout/vllm_rollout/web_search/web_search_main.py` with your search API:

```python
def deep_search_snippet(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="your bing api key", bing_endpoint="https://api.bing.microsoft.com/v7.0/search"):
    args = Namespace(
        dataset_name='qa',
        split='test',
        subset_num=-1,
        max_search_limit=15,
        top_k=top_k,  
        use_jina=use_jina,  
        jina_api_key=jina_api_key,  
        temperature=0.7,
        top_p=0.8,
        min_p=0.05,
        top_k_sampling=20,
        repetition_penalty=1.05,
        max_tokens=4096,
        bing_subscription_key=bing_subscription_key, 
        bing_endpoint=bing_endpoint, 
        eval=False,
        seed=1742208600,
        concurrent_limit=200
    )
```

Replace `bing_subscription_key`, `bing_endpoint`, and `api_base_url` with your own values. Various web search modes are provided in this file for you to choose from.

You can then run the following script to start training:

```bash
cd ./Tool_Star_RL/scripts/train/
bash run_tool_star.sh
```

For the core code of the rollout process, please refer to `/src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`, and for the reward calculation part, refer to `/Tool_Star_RL/src/verl/verl/utils/reward_score`. You can modify them according to your needs.
