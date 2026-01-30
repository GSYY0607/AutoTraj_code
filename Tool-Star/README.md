
<div align="center"> 
<img src="https://github.com/dongguanting/Tool-Star/blob/main/img/image.png" width="120px">
</div>



<h1 align="center"> 🔧✨Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning</a></h1>


<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2505.16410)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/2505.16410)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/_akhaliq/status/1925924431676821698)
</div>

<p align="center">
🤗 <a href="https://huggingface.co/dongguanting/Tool-Star-Qwen-0.5B" target="_blank">Tool-Star-Qwen-0.5B</a> ｜
🤗 <a href="https://huggingface.co/dongguanting/Tool-Star-Qwen-1.5B" target="_blank">Tool-Star-Qwen-1.5B</a> ｜
🤗 <a href="https://huggingface.co/dongguanting/Tool-Star-Qwen-3B" target="_blank">Tool-Star-Qwen-3B</a> ｜
🤗 <a href="https://huggingface.co/dongguanting/Tool-Star-Qwen-7B" target="_blank">Tool-Star-Qwen-7B</a> ｜
  </p>
  <p align="center">
🤗 <a href="https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K" target="_blank">Tool-Star-SFT-54K</a> ｜
🤗 <a href="https://huggingface.co/datasets/dongguanting/Multi-Tool-RL-10K" target="_blank">Multi-Tool-RL-10K</a>
</p>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
- **[Oct 16, 2025]**: 🚀🚀🚀 We propose a new algorithm [**AEPO**](https://www.arxiv.org/abs/2510.14545), which focused on entropy-balanced agentic RL and consistently outperforms ARPO on datasets like GAIA, HLE, and AIME. Full [codebase](https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO) and [🤗 HF-Models](https://huggingface.co/collections/dongguanting/aepo-68ef6832c99697ee03d5e1c7) of AEPO released.
- **[July 25, 2025]**: 🚀🚀🚀 We have released a new project **[ARPO](https://github.com/dongguanting/ARPO)** , which significantly accelerates the training process for Tool-star (**~4 times faster** ) and supports training for the Qwen2.5, Qwen3, and Llama3 series models! We welcome everyone to try and star it!!
- **[June 30, 2025]**: 🔥 We have updated our **[🤗Tool-Star-Qwen-7B](https://huggingface.co/dongguanting/Tool-Star-Qwen-7B)** and refreshed the **[Performance of Tool-Star Series Models](#-performance-of-tool-star-models)** in the README. We welcome everyone to reproduce and cite it!
- **[June 6, 2025]**: We released more lightweight checkpoints of Tool-Star . Checkout **[🤗Tool-Star-Qwen-0.5B](https://huggingface.co/dongguanting/Tool-Star-Qwen-0.5B)** & **[🤗Tool-Star-Qwen-1.5B](https://huggingface.co/dongguanting/Tool-Star-Qwen-1.5B)** here.
- **[May 21, 2025]**: The brief introduction of Tool-Star can be found on platforms like **[X](https://x.com/_akhaliq/status/1925924431676821698), [Zhihu](https://zhuanlan.zhihu.com/p/1911573573602115645) and [Wechat](https://mp.weixin.qq.com/s/UNP3P2GEtIuYhT7Z8wIV1g?scene=1)**.
- **[May 21, 2025]**: **[🤗 Tool-Star Collection](https://huggingface.co/collections/dongguanting/tool-star-682fd73dfa508bf3f40da032)** is now available on Hugging Face. We will keep update it!
- **[May 21, 2025]**: 🔥 We released an our cold-star SFT and RL dataset for tool-integrated reasoning. Checkout **[🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K)** and **[Multi-Tool-RL-10K](https://huggingface.co/datasets/dongguanting/Multi-Tool-RL-10K)** here.
- **[May 21, 2025]**: 🔥 We released our Tool-Star-Qwen-3B checkpoint. Checkout **[🤗Tool-Star-Qwen-3B](https://huggingface.co/dongguanting/Tool-Star-Qwen-3B)** here.
- **[May 21, 2025]**: 📄 Our paper is now available on **[arXiv](https://arxiv.org/pdf/2505.16410)** and **[Hugging Face](https://huggingface.co/papers/2505.16410)** daily paper.
- **[May 21, 2025]**: 🚀 Full codebase released. Tool-Star supports multiple Tools with several open-source models like Qwen2.5-3B-Instruct.



## 🔥 Agentic RL Family

<details open><summary>👏 Welcome to try our agentic RL series of algorithms: </summary><p>
<!--  may -->


> [**Agentic Entropy-Balanced Policy Optimization**]() <br>
> **Authors:** Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou†, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou† <br>
> **TLDR:** An agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/ARPO) [![github](https://img.shields.io/github/stars/RUC-NLPIR/ARPO.svg?style=social)](https://github.com/RUC-NLPIR/ARPO) [![arXiv](https://img.shields.io/badge/Arxiv-2510.14545-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.14545) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2510.14545) [![Collection](https://img.shields.io/badge/Hugging%20Face-Dataset%20&%20Models-yellow?logo=huggingface)](https://huggingface.co/collections/dongguanting/aepo-68ef6832c99697ee03d5e1c7) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)]()


> [**Agentic Reinforced Policy Optimization**](https://arxiv.org/abs/2507.19849) <br>
> **Authors:** Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao , Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou†, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou† <br>
> **TLDR:** An agentic RL algorithm encourage the policy model to adaptively branch sampling during high-entropy tool-call rounds, <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/ARPO) [![github](https://img.shields.io/github/stars/RUC-NLPIR/ARPO.svg?style=social)](https://github.com/RUC-NLPIR/ARPO) [![arXiv](https://img.shields.io/badge/Arxiv-2507.19849-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.19849) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2507.19849) [![Collection](https://img.shields.io/badge/Hugging%20Face-Dataset%20&%20Models-yellow?logo=huggingface)](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/_akhaliq/status/1950172418250547478)

> [**Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning**](https://arxiv.org/abs/2505.16410) <br>
> **Authors:** Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou†, Ji-Rong Wen <br>
> **TLDR:** An end-to-end TIR post-training framework that empowers LLMs to autonomously interact with multi-tool environments through Self-Critic RL design<br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/Tool-Star) [![github](https://img.shields.io/github/stars/RUC-NLPIR/Tool-Star.svg?style=social)](https://github.com/RUC-NLPIR/Tool-Star) [![arXiv](https://img.shields.io/badge/Arxiv-2505.16410-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.16410) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2505.16410) [![Collection](https://img.shields.io/badge/Hugging%20Face-Dataset%20&%20Models-yellow?logo=huggingface)](https://huggingface.co/collections/dongguanting/tool-star-682fd73dfa508bf3f40da032) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/_akhaliq/status/1925924431676821698)
</p></details>



## :mag_right: Roadmap

Tool-star is still under development and there are many issues and room for improvement. We will continue to update. And we also sincerely welcome contributions on this open-source toolkit.
- [x] Release tiny LLM version (e.g. 0.5B, 1.5B)
- [x] Support larger parameter size LLM (e.g. 7B)
- [x] Update asynchronous and efficient training framework.（See **[ARPO](https://github.com/dongguanting/ARPO)** , which significantly accelerates the training process for Tool-star (**~4 times faster** )）

## Table of Contents

- [Tool-Star](#🔧✨tool-star-empowering-llm-brained-multi-tool-reasoner-via-reinforcement-learning)
  - [Overall Performance](#-overall-performance)
- [Quick Start](#-quick-start-for-training)
  - [Cold-Start SFT Stage](#-cold-start-sft-stage)
    - [Environment Setup](#1-environment-setup)
    - [Fine-Tuning Model](#2-fine-tuning-model)
  - [Self-Critic RL Stage](#-self-critic-rl-stage)
    - [Environment Setup](#1-environment-setup-1)
    - [Vanilla RL Training](#2-vanilla-rl-training)
    - [Optional: Self-Critic DPO Training](#3-self-critic-dpo-training-optional)
  - [TIR Evaluation](#-tir-evaluation)
    - [Environment Setup](#1-environment-setup-2)
    - [LLM Service Deployment](#2-llm-service-deployment)
    - [Retriever Serving Deployment](#3-retriever-serving-deployment)
    - [Inference Your Model](#4-inference-your-model)
    - [Calculate Metrics](#5-calculate-metrics)
  - [Performance of Tool-Star Models](#-performance-of-tool-star-models)
- [Citation](#-citation)




## 💡 Overview


**Tool-Star** is a **reinforcement learning-based framework** designed to empower LLMs to autonomously invoke **multiple external tools** during stepwise reasoning. Specifically, Tool-Star integrates six types of tools into the reasoning process (three for training and three for inference-time optimization) and incorporates systematic designs in both data synthesis and training algorithms.

<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/edb21f58-7a18-47b9-8b1c-9522f7c9c56d" />
</p>

---

### 📊 Overall Performance
As shown below, Tool-Star demonstrates strong overall reasoning performance across more than **10** challenging computational reasoning tasks (e.g., AIME24 and MATH500) and knowledge-intensive reasoning tasks (e.g., WebWalker and HotpotQA), while ensuring both efficiency and reliability in tool usage.


<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/5f60be15-6992-413e-a405-a72e3b352fc1" />
</p>




# 🏃 Quick Start for Training

## ❄️ Cold-Start SFT Stage

### 1. Environment Setup

In this step, we will describe how to perform a cold start for the SFT stage using the Llama Factory repository. Please first set up the environment for [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

### 2. Fine-Tuning Model


1. Download your SFT dataset from [🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K) and place it in `LLaMA-Factory-main/data/final_sft_edition9.json`. Define the dataset in `dataset_info.json`.

2. Complete the path information in `LLaMA-Factory-main/examples/train_full/qwen_sft_tool_star.yaml`. The file content should be as follows:

```yaml
### model
model_name_or_path: {your_path_to_model}/Qwen2.5-3B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: final_sft_edition9
template: qwen
cutoff_len: 15000
max_samples: 1000000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {your_save_path}/Qwen2.5-3B-Instruct-final_sft_edition10-52
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 7.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

After completing the information, you can fine-tune the model using the following command:

```python
cd LLaMA-Factory-main
bash ./examples/train_full/train_sft.sh
```

---

## 🔥 Self-Critic RL Stage

In this step, we will load the cold-start data for GRPO training. We reference the [ReCall](https://github.com/Agent-RL/ReCall) and [VERL](https://github.com/volcengine/verl) frameworks for RL training.


### 1. Environment Setup

 you can install our additional environment as follow: 

```bash
#create env
conda env create -f environment.yaml
conda activate toolstar

# install flash-atten
pip3 install flash-attn --no-build-isolation

# install RL basic env
cd Tool_Star_RL
pip3 install -e .
```
Please refer to [requirements.txt](https://github.com/dongguanting/Tool-Star/blob/main/requirements.txt) carefully. It is important to note that **vLLM<= 0.6.3 and torch==2.4.0 (seem versions will not work.)**. You can also install a compatible flash_attention package from [here](https://github.com/Dao-AILab/flash-attention/releases).

If you encounter ray or other RL environment issues, we **highly recommend that you first try to run the RL training code for [ReCall](https://github.com/Agent-RL/ReCall/tree/re-search) or [Verl](https://github.com/volcengine/verl) successfully**, then further aligning with our [requirements.txt](https://github.com/dongguanting/Tool-Star/blob/main/requirements.txt).



### 2. Vanilla RL Training

Our training framework is based on [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). The training scripts can be found under `scripts/train`. First, you need to complete the information in `scripts/train/run_tool_star.sh`, 
we have provided both [train parquet](https://huggingface.co/datasets/dongguanting/Multi-Tool-RL-10K) and [test parquet](https://github.com/dongguanting/Tool-Star/blob/main/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet) for RL:

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

For the trained RL checkpoint, you can follow the code below to convert the weights to Hugging Face format：
```bash
# Merge RL weights and save in the same path.
python /Tool_Star_RL/model_merger.py \
    --local_dir /{your_checkpoint_path}/global_step_{your_RL_step}/actor/ \
```


### 3. Self-Critic DPO Training (Optional)

In our experiments, completing SFT + Vanilla RL has been sufficient to almost reproduce Tool-Star's performance (refer to the ablation study).

If you wish to proceed with Self-Critic DPO training, please refer to the training algorithm in **Appendix B.1** of the paper and the data format process in **Appendix E.2**. You can self-sample reward data using the saved checkpoints for RL and SFT training data. We also provide DPO training code based on [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) for your reference.

Please complete the path information in `LLaMA-Factory-main/examples/train_lora/qwen_lora_dpo_2.yaml` and place the synthesized DPO data in `LLaMA-Factory-main/data/`. You can then run the following script for training:

```bash
cd LLaMA-Factory-main
bash ./examples/train_lora/train_dpo.sh
```

---

## ✅ TIR Evaluation

If you have already trained a model, you can refer to the following process for TIR capability evaluation. Of course, you can also download our checkpoint **[🤗Tool-Star-Qwen-3B](https://huggingface.co/dongguanting/Tool-Star-Qwen-3B)** for directly testing.

### 1. Environment Setup

```bash
#create env
conda env create -f environment.yaml
conda activate toolstar

# install flash-atten
pip3 install flash-attn --no-build-isolation
```

### 2. LLM Service Deployment

In this step, we will use the VLLM framework to deploy additional large language models (LLMs). This includes deploying an LLM as a judging model to evaluate the accuracy of the generated answers in the subsequent steps, as well as deploying inference-time tools such as code debugging and chain refinement.

- We use Qwen2.5-72B-Instruct as the judging model.

- We use Qwen2.5-3B-Instruct, which has the same parameter scale as the base model, as the foundation for the inference-time tools.

For the specific deployment, you can refer to the following script.

```bash
cd evaluation
bash vllm_server.sh
```

### 3. Retriever Serving Deployment

In this section, we will deploy the retriever for performing search tasks on Wikipedia-based datasets. We provide a Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need to download the [pre-indexed Wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [Wikipedia corpus, and corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). The corpuses used can be found [here](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus), and Index construction method can be found [here](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

More details can be found in the [FlashRAG documentation](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

To start the retriever serving, first fill in `evaluation/search/serving_config.yaml` with the correct paths to the retrieval model, index, and corpus, as well as available GPU IDs. Then, run the following command to start the retriever serving:

```bash
cd evaluation/search
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```

### 4. Inference Your Model

In this section, we infer answers using a trained model. We support five types of mathematical reasoning datasets: AIME24, AIME25, GSM8K, MATH, and MATH500, as well as seven QA reasoning datasets: WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE. Due to resource constraints, all models and baselines will test a maximum of 500 samples for mathematical reasoning, 200 samples for all QA datasets, and 500 samples for HLE (please refer our code).

First, replace the API_URL and API key with your own in the following files:

In `evaluation/utils.py`:

```python
def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'your_search_api_url'
    ...

def batch_search(query: Union[str, List[str]], top_n=5) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'your_search_api_url'
    ...
```

In `evaluation/tools/web_search_main.py`:

```python
def deep_search(search_query, top_k=10, use_jina=False, jina_api_key="empty", bing_subscription_key="xxxxx", bing_endpoint="xxxxx/search"):
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
        api_base_url='xxxxx',  
        model_name='search-agent',
        concurrent_limit=200
    )
    ...
```

In `evaluation/tools/debug_code.py`:

```python
def debug_code_function(code, error, api_key="your_api_key"):

    API_BASE_URL = api_key
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    ...
```

Then, start the inference. We recommend that you use the default parameters as:

```bash
cd evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/path/to/your_path:$PYTHONPATH
module load cuda/11.8
python run.py \
    --model_path /path/to/your_model_path \
    --dataset_name math \
    --task math \
    --gpu_use 0.95 \
    --max_tokens 16384 \ #you can change this, 8192 is enough for most tasks
    --max_input_len 16384 \ #you can change this, 8192 is enough for most tasks
    --output_path /path/to/your_results/your_exp_math_result.json \
    --counts 500 \
    --batch_size 100 \
    --use_debug 
```

**Parameter Explanations:**
- `--model_path`: Path to your model.
- `--dataset_name`: Name of your dataset (supports AIME24, AIME25, GSM8K, MATH, MATH500, WebWalker, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue, GAIA, and HLE).
- `--task`: Set to `math` for mathematical reasoning datasets and `qa` for QA reasoning datasets.
- `--gpu_use`: GPU memory utilization.
- `--max_tokens`: Maximum number of tokens the model can generate.
- `--max_input_len`: Maximum input tokens the model can accept.
- `--output_path`: Path to save the results.
- `--counts`: Number of samples to take from the test set during testing.
- `--batch_size`: Batch size for parallel inference.
- `--use_debug`: Enable the debug mechanism.

**Additional Parameters（Optional）:**

In practical, only in the cases of HLE and GAIA is there a possibility of exceeding the length limit, you can use refiner. Generally, it won't occur in other situations.  

- `--use_rollback`: Whether to use the rollback mechanism.
- `--use_refiner`: Whether to use the refine mechanism.


In `evaluation/tools/refine_code.py`:

```python
def refine(prompt, response):

    API_BASE_URL = "your_api_base_url"
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    ...
```

### 5. Calculate Metrics

First, replace the API URL and API key with your own in the following file:

In `evaluation/evaluate/scripts/evaluate.py`:

```python
async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = "http://114514.1919810/v1"
    if model_name is None:
        model_name = "Qwen2.5-72B-Instruct"
    ...
```

Replace `api_base_url` with the API_URL of your deployed model.

Then, run the following command:

```bash
cd evaluation
python evaluate/scripts/evaluate.py \
    --output_path /path/to/your_results/your_exp_math_result.json \
    --task math \
    --dataset_name math \
    --use_llm \
    --extract_answer
```

**Parameter Explanations:**
- `--output_path`: Path to save the results.
- `--task`: Set to `math` for mathematical reasoning datasets and `qa` for QA reasoning datasets.
- `--dataset_name`: Name of your dataset.
- `--use_llm`: Whether to use the LLM-as-judge mechanism.
- `--extract_answer`: Whether to use exact matching (removes \text and other redundant symbols).

---

## 📄 Performance of Tool-Star Models

We present the results of our Tool-Star model checkpoints with sizes 0.5B, 1.5B, 3B, and 7B, all based on the Qwen2.5-Instruct series. The results of **“Self-Critic-RL”** setting correspond to our series of 🤗 open-source huggingface model checkpoints.

<p align="center">
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/108d51c1-d4ff-4045-b9cb-5829dded97da" />
</p>




## 📄 Citation

If you find this work helpful, please cite our papers:

```bibtex
@article{dong2025tool,
  author       = {Guanting Dong and
                  Yifei Chen and
                  Xiaoxi Li and
                  Jiajie Jin and
                  Hongjin Qian and
                  Yutao Zhu and
                  Hangyu Mao and
                  Guorui Zhou and
                  Zhicheng Dou and
                  Ji{-}Rong Wen},
  title        = {Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement
                  Learning},
  journal      = {CoRR},
  volume       = {abs/2505.16410},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.16410},
  doi          = {10.48550/ARXIV.2505.16410},
  eprinttype    = {arXiv},
  eprint       = {2505.16410},
  timestamp    = {Thu, 26 Jun 2025 07:49:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-16410.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


@article{dong2025arpo,
  author       = {Guanting Dong and
                  Hangyu Mao and
                  Kai Ma and
                  Licheng Bao and
                  Yifei Chen and
                  Zhongyuan Wang and
                  Zhongxia Chen and
                  Jiazhen Du and
                  Huiyang Wang and
                  Fuzheng Zhang and
                  Guorui Zhou and
                  Yutao Zhu and
                  Ji{-}Rong Wen and
                  Zhicheng Dou},
  title        = {Agentic Reinforced Policy Optimization},
  journal      = {CoRR},
  volume       = {abs/2507.19849},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2507.19849},
  doi          = {10.48550/ARXIV.2507.19849},
  eprinttype    = {arXiv},
  eprint       = {2507.19849},
  timestamp    = {Fri, 22 Aug 2025 07:48:19 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2507-19849.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


@article{dong2025aepo,
  author       = {Guanting Dong and
                  Licheng Bao and
                  Zhongyuan Wang and
                  Kangzhi Zhao and
                  Xiaoxi Li and
                  Jiajie Jin and
                  Jinghan Yang and
                  Hangyu Mao and
                  Fuzheng Zhang and
                  Kun Gai and
                  Guorui Zhou and
                  Yutao Zhu and
                  Ji{-}Rong Wen and
                  Zhicheng Dou},
  title        = {Agentic Entropy-Balanced Policy Optimization},
  journal      = {CoRR},
  volume       = {abs/2510.14545},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.14545},
  doi          = {10.48550/ARXIV.2510.14545},
  eprinttype    = {arXiv},
  eprint       = {2510.14545},
  timestamp    = {Fri, 14 Nov 2025 15:17:45 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2510-14545.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}



```

## 🤝 Acknowledge

This training implementation builds upon [Llama Factory](https://github.com/hiyouga/LLaMA-Factory), [verl](https://github.com/volcengine/verl) and [ReCall](https://github.com/Agent-RL/ReCall). For evaluation, we rely on [WebThinker](https://github.com/RUC-NLPIR/WebThinker), [Search-o1](https://github.com/sunnynexus/Search-o1), and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). The Python interpreter design references [ToRA](https://github.com/microsoft/ToRA) and [ToRL](https://github.com/GAIR-NLP/ToRL), while our models are trained using [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/). We express our sincere gratitude to these projects for their invaluable contributions to the open-source community. 


## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [dongguanting@ruc.edu.cn](dongguanting@ruc.edu.cn).


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dongguanting/Tool-Star&type=Date)](https://www.star-history.com/#dongguanting/Tool-Star&Date)
