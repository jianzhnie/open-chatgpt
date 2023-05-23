nohup sh scripts/run_alpaca_ds.sh > run2.log 2>&1 &
nohup sh scripts/run_alpaca_lora.sh > run2.log 2>&1 &


python generate.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  \
    --temperature 1.0 \
    --repetition_penalty 1.0 \
    --top_k  40 \
    --top_p  0.9  \
    --num_beams 4 \
