#!/bin/bash
python convert_llama_weights_to_hf.py \
    --input_dir /userhome/LLM_checkpoints --model_size 7B --output_dir /userhome/jianzhnie/llama-checkpoint/7B

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1  --nproc_per_node 4 train_supervised_finetune.py \
         --model_path=/userhome/jianzhnie/llama-checkpoint/7B  \
         --streaming --no_gradient_checkpointing \
        --learning_rate 1e-5  \
        --max_steps 5000 \
        --output_dir ./llama-se


accelerate launch --multi_gpu --num_machines 1  --num_processes 8 \
        examples/stack_llama/scripts/rl_training.py \
        --log_with=wandb  \
        --model_name=/userhome/jianzhnie/llama-checkpoint/7B \
        --reward_model_name=<LLAMA_SE_RM_MODEL> \
        --adafactor=False \
        --tokenizer_name=<LLAMA_TOKENIZER> \
        --save_freq=100 \
        --output_max_length=128 \
        --batch_size=8 \
        --gradient_accumulation_steps=8  \
        --batched_gen=True \
        --ppo_epochs=4 \
        --seed=0 \
        --learning_rate=1.4e-5 \
        --early_stopping=True \
        --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam


python train_supervised_finetune.py \
         --model_path=facebook/opt-125m \
         --streaming --no_gradient_checkpointing \
        --learning_rate 1e-5  \
        --no_fp16 \
        --max_steps 5000 \
        --output_dir ./llama-se