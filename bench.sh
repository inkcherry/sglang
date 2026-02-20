#!/bin/bash
set -ex

#git clone https://github.com/kimbochen/bench_serving.git

# Run benchmark
max_concurrency=${1:-256}
max_concurrency=1
input_len=${2:-1024}
output_len=${3:-16}

random_range_ratio=1
num_prompts=$((max_concurrency * 3))


# MODEL_NAME="/apps/data/models/DSR1"
# # MODEL_NAME="/models/DSR1-MXFP4"
# MODEL_NAME="/apps/data/models/DSV3"
MODEL_NAME="/apps/data/models/DSR1-0528"

#infernce max

# python3 bench_serving/benchmark_serving.py \
#             --model $MODEL_NAME \
#                 --backend openai \
#                     --base-url "http://0.0.0.0:8000" \
#                         --dataset-name random \
#                             --random-input-len "$input_len" \
#                                 --random-output-len "$output_len" \
#                                     --random-range-ratio "$random_range_ratio" \
#                                         --num-prompts "$num_prompts" \
#                                             --max-concurrency "$max_concurrency" \
#                                                 --request-rate inf \
#                                                     --ignore-eos --use-chat-template

# sglang

python -m sglang.bench_serving \
    --backend sglang \
    --model /apps/data/models/DSR1-0528 \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio "$random_range_ratio" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$max_concurrency" \
    --request-rate inf \
    --port 8000 \
    --disable-ignore-eos \
    --apply-chat-template \
    --profile

# python -m sglang.bench_serving --backend sglang --model /apps/data/models/DSR1-0528 --num-prompts 256 --sharegpt-output-len 10 --port 8888 --profile
