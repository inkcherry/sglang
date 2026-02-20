#!/bin/bash

export GLOO_SOCKET_IFNAME=enp81s0f1
export NCCL_SOCKET_IFNAME=enp81s0f1
export SGLANG_USE_AITER=1
# export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles

# export SGLANG_MORI_FP8_DISP=True
# export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
export SGLANG_TORCH_PROFILER_DIR=/apps/mingzliu/profile_tp
export SGLANG_PROFILE_WITH_STACK=True
export SGLANG_PROFILE_RECORD_SHAPES=True


# python3 -m sglang.launch_server \
#     --model-path /apps/data/models/DeepSeek-R1  \
#     --tp-size 8 \
#     --decode-log-interval 1 \
#     --host 0.0.0.0 \
#     --port 8888 \
#     --trust-remote-code \
#     --watchdog-timeout 1000000 \
#     --mem-fraction-static 0.6 \
#     --max-running-requests 256 \
#     --chunked-prefill-size 8196 \
#     --speculative-algorithm EAGLE \
#     --speculative-num-steps 1 \
#     --speculative-eagle-topk 1 \
#     --speculative-num-draft-tokens 2 \
#     --kv-cache-dtype fp8_e4m3 \
#     --attention-backend aiter \
#     --disable-cuda-graph \

    


python3 -m sglang.launch_server \
    --model-path /apps/data/models/DSR1-0528  \
    --tp-size 8 \
    --decode-log-interval 40 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --watchdog-timeout 1000000 \
    --mem-fraction-static 0.6 \
    --max-running-requests 256 \
    --chunked-prefill-size 8196 \
    --speculative-algorithm EAGLE  \
    --speculative-num-steps 3  \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --kv-cache-dtype fp8_e4m3  \
    --attention-backend aiter  \
    --cuda-graph-bs $(seq 1 8) \
     --enable-aiter-allreduce-fusion
    # --disable-cuda-graph 

#--log-reques


    #   --piecewise-cuda-graph-compiler eager\



 
        # --cuda-graph-bs $(seq 1 256) \

 


 # python -m sglang.bench_serving --backend sglang --model /apps/data/models/DeepSeek-R1 --num-prompts 10 --sharegpt-output-len 10 --port 8888 --profile
# python -m sglang.bench_serving --backend sglang --model /apps/data/models/DeepSeek-R1 --num-prompts 10 --sharegpt-output-len 10 --port 8888 --profile