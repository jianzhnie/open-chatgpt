CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main.py \
   --data_path Dahoas/rm-static \
   --model_name_or_path /userhome/jianzhnie/checkpoints/llama-checkpoint/7B/ \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 16  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir work_dirs \
   &> work_dirs/training.log
