#!/bin/bash
# 脚本名称：start_generate.sh
# 功能：一键启动轨迹生成脚本，解决CUDA多进程冲突

# =========================
# 核心环境变量配置（关键）
# =========================
# 强制vLLM子进程用spawn启动（解决CUDA重初始化错误）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 限制CPU线程数，避免CPU竞争
export OMP_NUM_THREADS=1
# 明确指定可用GPU（根据实际显卡数量调整，如0,1,2,3）
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 禁用Ray分布式框架
export VLLM_USE_RAY=0
# 禁用CUDA graph，强制eager模式
export VLLM_ENFORCE_EAGER=1
# 避免tokenizer并行警告
export TOKENIZERS_PARALLELISM=false
# 优化CUDA内存分配
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# =========================
# 执行Python脚本
# =========================
echo "🚀 开始启动轨迹生成脚本..."
echo "📌 输入数据：/to/your/path/input"
echo "📌 输出数据：/to/your/path/output"
echo "📌 模型路径：/to/your/path/model"

# 执行Python脚本
/to/your/path/inference/bin/python \
  /to/your/path/generate_trajectories_final.py
