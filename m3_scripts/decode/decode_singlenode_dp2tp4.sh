# 环境变量设置
nic_name="enp35s0f2"
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name  

export HCCL_BUFFSIZE=256
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false  
export OMP_NUM_THREADS=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 # VLLM框架超时时间
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export TASK_QUEUE_ENABLE=1
# profiling 
export VLLM_TORCH_PROFILER_DIR="/home/z00946994/profiling"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export ASCEND_LAUNCH_BLOCKING=0
# for sparse attention
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.T560/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.T560/opp/vendors/custom_transformer/op_api/lib/:$ASCEND_CUSTOM_OPP_PATH
export PATH=/home/g00893696/ascendnpu-ir/tools/bishengir/bin:$PATH

# flash_comm
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
# mega moe
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0

export VLLM_ASCEND_MOE_DEBUG_DISTRIBUTE_BARRIER=1
# export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
# export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
# export HCCL_OP_EXPANSION_MODE="CCU_SCHED"

vllm serve /home/g00893696/weight/MiniMax-M3-MXFP8/  \
     --host 141.61.94.58 \
     --port 10086 \
     --served-model-name minimax \
     --trust-remote-code \
     --dtype bfloat16 \
     --max-num-seqs 32 \
     --max-num-batched-tokens 2048 \
     --max-model-len 150000 \
     --tensor-parallel-size 4 \
     --data-parallel-size 2 \
     --gpu-memory-utilization 0.92 \
     --distributed_executor_backend "mp" \
     --no-enable-prefix-caching \
     --reasoning-parser minimax_m3 \
     --enable-expert-parallel \
     --quantization ascend \
     --api-server-count=1 \
     --kv-cache-dtype fp8 \
     --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
     --safetensors-load-strategy 'prefetch' \
     --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false, "fuse_norm_quant":false}, "indexer_kv_dtype": "fp8", "multistream_overlap_shared_expert": true}' \
     --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/x30075441/prof/minimax_486t_deocde_sp_score_fp8_729", "torch_profiler_with_stack": false}' \
     > decode_log/minimax_486t_decode_128k_bs8_0724_force3.log 2>&1 &

#          --hf-overrides '{"text_config": {"num_hidden_layers": 30}}' \
