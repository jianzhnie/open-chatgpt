<div align="center">
  <img src="assets/logo.png" width="800"/>
<div>&nbsp;</div>
</div>

<div align="center">

[中文](README_zh.md) | English
</div>

# Open-ChatGPT: A Chatbot Based on Llama Model

![GitHub Stars](https://img.shields.io/github/stars/jianzhnie/open-chatgpt.svg?label=Stars&style=social)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/jianzhnie/open-chatgpt/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Table of Contents
- [Open-ChatGPT: A Chatbot Based on Llama Model](#open-chatgpt-a-chatbot-based-on-llama-model)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [☕ Quick Start ☕](#-quick-start-)
  - [Fintune Alpaca](#fintune-alpaca)
    - [Training (`finetune.py`)](#training-finetunepy)
    - [Using DeepSpeed](#using-deepspeed)
  - [PEFT(Parermeter Efficient Fine-Tuning)](#peftparermeter-efficient-fine-tuning)
    - [Inference (`generate.py`)](#inference-generatepy)
  - [Server](#server)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)


## Introduction
This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
We provide an Instruct model of similar quality to `text-davinci-003` that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research),
and the code is easily extended to the `13b`, `30b`, and `65b` models.

In addition to the training code, which runs within hours on a single RTX 4090,
we publish a script for downloading and inference on the foundation model and LoRA,
as well as the resulting [LoRA weights themselves](https://huggingface.co/tloen/alpaca-lora-7b/tree/main).
To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.


## ☕ Quick Start ☕

```bash
git clone https://github.com/jianzhnie/open-chatgpt.git
pip install -r requirements.txt
```

## Fintune Alpaca


### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

Example usage:

```bash
python train_alpaca.py \
    --model_name_or_path  'decapoda-research/llama-7b-hf' \
    --data_path tatsu-lab/alpaca  \
    --output_dir work_dir/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1
```


### Using DeepSpeed

Naively, fine-tuning a 7B model requires about 7 x 4 x 4 = 112 GB of VRAM. Commands given above enable parameter sharding, so no redundant model copy is stored on any GPU.
If you'd like to further reduce the memory footprint, here are some options:

- Turn on CPU offload for FSDP with `--fsdp "full_shard auto_wrap offload"`. This saves VRAM at the cost of longer runtime.
- In our experience, DeepSpeed stage-3 (with offload) can at times be more memory efficient than FSDP with offload. Here's an example to use DeepSpeed stage-3 with 4 GPUs with both parameter and optimizer offload:

```bash
pip install deepspeed
torchrun --nproc_per_node=8 train_alpaca.py \
    --model_name_or_path  'decapoda-research/llama-7b-hf' \
    --data_path tatsu-lab/alpaca  \
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
```

- [LoRA](https://arxiv.org/abs/2106.09685) fine-tunes low-rank slices of the query, key, and value embedding heads. This can reduce the total memory footprint from 112GB to about 7x4=28GB. We may release our re-implemention of this in the future, but for now the [peft](https://github.com/huggingface/peft) codebase can be a useful resource.


## PEFT(Parermeter Efficient Fine-Tuning)


```bash
python train_alpaca_lora.py \
    --model_name_or_path  decapoda-research/llama-7b-hf  \
    --data_path tatsu-lab/alpaca  \
    --output_dir work_dir_lora/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

## Server

```bash
python generate_server.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  tloen/alpaca-lora-7b \
    --load_8bit
```

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License

`Openn-ChatGPT` is released under the Apache 2.0 license.

## Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{open-chatgpt,
  author = {jianzhnie},
  title = {Open-ChatGPT, a chatbot based on Llama model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jianzhnie/open-chatgpt}},
}
```
