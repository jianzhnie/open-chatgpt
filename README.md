<div align="center">
  <img src="assets/logo.png" width="800"/>
<div>&nbsp;</div>
</div>

<div align="center">

[中文](README_zh.md) | English
</div>

# Open-ChatGPT: An open-source implementation of ChatGPT


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/jianzhnie/open-chatgpt/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction

`Open-ChatGPT` is a open-source library that allows you to train a hyper-personalized ChatGPT-like ai model using your own data and the least amount of compute possible.

`Open-ChatGPT` is a general system framework for enabling an end-to-end training experience for ChatGPT-like models. It can automatically take your favorite pre-trained large language models though an OpenAI InstructGPT style three stages to produce your very own high-quality ChatGPT-style model.

We have Impleamented RLHF (Reinforcement Learning with Human Feedback) powered by transformer library and DeepsSpeed. It supports distributed training and offloading, which can fit extremly large models.

If you like the project, please show your support by [leaving a star ⭐](https://github.com/jianzhnie/open-chatgpt/stargazers).

## News

- [2023/05] 🔥 We implement **Stanford Alpaca Lora**.

- [2023/05] 🔥 We implement **Stanford Alpaca**.
- [2023/04] We released **RLHF(Reinforcement Learning with Human Feedback)  Pipeline** .
- [2023/03] We released the code **OpenChatGPT An Open-Source libraray to train ChatBot like ChatGPT **.

## Table of Contents

- [Open-ChatGPT: An open-source implementation of ChatGPT](#open-chatgpt-an-open-source-implementation-of-chatgpt)
  - [Introduction](#introduction)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Instruction Data](#instruction-data)
  - [Install](#install)
  - [Fintune](#fintune)
    - [Fine-tuning Alpaca-7B](#fine-tuning-alpaca-7b)
    - [Using DeepSpeed](#using-deepspeed)
    - [Fine-tuning Alpaca-7B with Lora](#fine-tuning-alpaca-7b-with-lora)
  - [Inference](#inference)
    - [No Enough Memory](#no-enough-memory)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)


## Instruction Data

## Install

```bash
git clone https://github.com/jianzhnie/open-chatgpt.git
pip install -r requirements.txt
```

## Fintune

### Fine-tuning Alpaca-7B

We fine-tune our models using standard Hugging Face training code. We fine-tune LLaMA-7B and LLaMA-13B with the following hyperparameters:

| Hyperparameter | LLaMA-7B | LLaMA-13B |
| -------------- | -------- | --------- |
| Batch size     | 128      | 128       |
| Learning rate  | 2e-5     | 1e-5      |
| Epochs         | 3        | 5         |
| Max length     | 512      | 512       |
| Weight decay   | 0        | 0         |

You can use the following command to train Alpaca-7B with 4 x A100 (40GB).

```bash
cd examples/alpaca/
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

If you meet OOM error, consider this.

Naively, fine-tuning a 7B model requires about 7 x 4 x 4 = 112 GB of VRAM. Commands given above enable parameter sharding, so no redundant model copy is stored on any GPU.
If you'd like to further reduce the memory footprint, here are some options:

- Turn on CPU offload for FSDP with `--fsdp "full_shard auto_wrap offload"`. This saves VRAM at the cost of longer runtime.
- In our experience, DeepSpeed stage-3 (with offload) can at times be more memory efficient than FSDP with offload. Here's an example to use DeepSpeed stage-3 with 4 GPUs with both parameter and optimizer offload:

```bash
pip install deepspeed
cd examples/alpaca/
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

- [LoRA](https://arxiv.org/abs/2106.09685) fine-tunes low-rank slices of the query, key, and value embedding heads. This can reduce the total memory footprint from 112GB to about 7x4=28GB. 

### Fine-tuning Alpaca-7B with Lora

This part reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).


To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.


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

## Inference

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate_server.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path  tloen/alpaca-lora-7b 
```

### No Enough Memory

If you do not have enough memory, you can enable 8-bit compression by adding `--load-8bit` to commands above. This can reduce memory usage by around half with slightly degraded model quality. It is compatible with the CPU, GPU, and Metal backend. Alpaca-7B with 8-bit compression can run on a single NVIDIA 3090/4080/T4/V100(16GB) GPU.

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

## Acknowledgements

We appreciate the work by many open-source contributors, especially:

- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/)
- [LoRA](https://github.com/microsoft/LoRA/)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/)
- [Hugging Face](https://huggingface.co/)
- [LLaMa](https://github.com/facebookresearch/llama/)
- [Vicuna](https://github.com/lm-sys/FastChat/)

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
