#!/bin/sh
python generate.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  tloen/alpaca-lora-7b \
    --load_8bit \
    --temperature 0.9 \
    --repetition_penalty 1.0 \
    --top_k  40 \
    --top_p  0.75  \
    --num_beams 4


python generate_server.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  tloen/alpaca-lora-7b \
    --load_8bit

python generate_server.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  ./work_dir_lora \
    --load_8bit