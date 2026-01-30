CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/your_model_path \
    --served-model-name Qwen2.5-72B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --uvicorn-log-level debug \
    --host 0.0.0.0 \
    --port 114514