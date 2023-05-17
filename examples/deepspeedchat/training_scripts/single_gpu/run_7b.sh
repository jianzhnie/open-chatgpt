#!/bin/bash

nohup deepspeed --num_gpus 1 main.py \
    --data_path Dahoas/rm-static \
    --model_name_or_path facebook/opt-125m \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_seq_len 512 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --num_train_epochs 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage 0 \
    --lora_dim 128 \
    --lora_module_name decoder.layers. \
    --only_optimize_lora \
    --deepspeed \
    --output_dir work_dirs \
    &> work_dirs/training.log
