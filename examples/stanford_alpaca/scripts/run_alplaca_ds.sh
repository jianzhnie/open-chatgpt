CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15  torchrun --nproc_per_node=8 train_alpaca.py \
    --model_name_or_path /userhome/jianzhnie/checkpoints/llama-checkpoint/7B/ \
    --data_path ./alpaca_data.json \
    --output_dir work_dir/  \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "scripts/ds_config_zero3_auto.json"
    