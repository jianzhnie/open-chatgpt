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
- [2023/03] We released the code **OpenChatGPT: An Open-Source libraray to train ChatBot like ChatGPT**.

## Table of Contents

- [Open-ChatGPT: An open-source implementation of ChatGPT](#open-chatgpt-an-open-source-implementation-of-chatgpt)
  - [Introduction](#introduction)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Data Collection](#data-collection)
    - [Instruction Datasets](#instruction-datasets)
    - [RLHF Datasets](#rlhf-datasets)
    - [Data Preprocessing](#data-preprocessing)
    - [Data Fomatting](#data-fomatting)
  - [Install](#install)
  - [Instruction Fintune](#instruction-fintune)
    - [Fine-tuning Alpaca-7B](#fine-tuning-alpaca-7b)
    - [Using DeepSpeed](#using-deepspeed)
    - [Fine-tuning Alpaca-7B with Lora](#fine-tuning-alpaca-7b-with-lora)
  - [Inference](#inference)
    - [No Enough Memory](#no-enough-memory)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Data Collection

### Instruction Datasets

A collection of open-source instruction tuning datasets to train (text and multi-modal) chat-based LLMs (GPT-4, ChatGPT,LLaMA,Alpaca). 

Referring to [this](https://github.com/jianzhnie/awesome-instruction-datasets) ([@jianzhnie](https://github.com/jianzhnie)), we labeled each collected dataset according to the following rules:

(Lang)Lingual-Tags:

- EN: Instruction datasets in English
- CN: Instruction datasets in Chinese
- ML: [Multi-lingual] Instruction datasets in multiple languages

(Task)Task-Tags:

- MT: [Multi-task] Datasets containing multiple tasks
- TS: [Task-specific] Datasets tailored for specific tasks

(Gen)Generation-method:

- HG: [Human Generated Dataset] Datasets created by humans
- SI: [Self-Instruct] Datasets generated using self-instruct methods
- MIX: [Mixed Dataset] Dataset contains both human and machine generated data
- COL: [Collection of Dataset] Dataset made from a collection of other datasets

| Project                                                      |                           Datasets                           | Org                        | Nums    | Lang  | Task  | Gen  | Type                                      | Src                                 |
| :----------------------------------------------------------- | :----------------------------------------------------------: | -------------------------- | :------ | :---- | :---- | :--- | :---------------------------------------- | :---------------------------------- |
| [Chain of Thought](https://github.com/google-research/FLAN)  | [cot_data](https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data) \|[few_shot_data](https://github.com/google-research/FLAN/tree/main/flan/v2/niv2_few_shot_data) | Google                     | 74771   | EN/CN | MT    | HG   | instruct with cot reasoning               | annotating CoT on existing data     |
| [GPT4all](https://github.com/nomic-ai/gpt4all)               | [nomic-ai/gpt4all-j-prompt-generations](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations) | nomic-ai                   | 806199  | EN    | MT    | COL  | code, storys and dialogs                  | distillation from GPT-3.5-turbo     |
| [GPTeacher](https://github.com/teknium1/GPTeacher)           | [GPT-4 General-Instruct ](https://github.com/teknium1/GPTeacher/tree/main/Instruct)\|[Roleplay-Instruct](https://github.com/teknium1/GPTeacher/tree/main/Roleplay) \|[Code-Instruct ](https://github.com/teknium1/GPTeacher/tree/main/Codegen)\| [Toolformer](https://github.com/teknium1/GPTeacher/tree/main/Toolformer) | teknium1                   | 29013   | EN    | MT    | SI   | general, roleplay, toolformer             | GPT-4 & toolformer                  |
| [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | JosephusCheung             | 534610  | ML    | MT    | SI   | various linguistic tasks                  | text-davinci-003                    |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)    | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Hello-SimpleAI \| 万得资讯 | 37175   | EN/CN | TS    | MIX  | dialogue evaluation                       | human or ChatGPT                    |
| [HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | Hello-SimpleAI\|万得资讯   | 13k     | CN    | TS    | MIX  | dialogue evaluation                       | human or ChatGPT                    |
| [alpaca](https://github.com/tatsu-lab/stanford_alpaca)       | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | tatsu-lab                  | 52002   | EN    | MT    | SI   | general instruct                          | text-davinci-003                    |
| [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | yahma                      | 52k     | EN    | MT    | SI   | general instruct                          | text-davinci-003                    |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [alpaca_data_zh_51k](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/data/alpaca_data_zh_51k.json) | ymcui(讯飞)                | 51k     | CN    | MT    | SI   | general instruct                          | text-davinci-003                    |
| [Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM)  骆驼 | [trans_chinese_alpaca_data](https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/data/trans_chinese_alpaca_data.json) | LC1332(商汤)               | 52k     | CN    | MT    | SI   | general instruct                          | text-davinci-003                    |
| [Natural Instructions](https://github.com/allenai/natural-instructions) | [Allen AI 61 task](https://instructions.apps.allenai.org/#:~:text=Download%20Natural%2DInstructions%20%2D%20v1.1)\|[1.5k task](https://instructions.apps.allenai.org/#:~:text=Natural%2DInstructions%20%2D%20v2-,.,-x) | Allen AI                   | 5040134 | ML    | MT    | COL  | diverse nlp tasks                         | human annotated datasets collection |
| [belle_cn](https://huggingface.co/BelleGroup)                | [BelleGroup/train_1M_CN](https://huggingface.co/datasets/bellegroup/train_1M_CN) \|[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/bellegroup/train_0.5M_CN) | BelleGroup(链家)           | 1079517 | CN    | TS/MT | SI   | general, mathematical reasoning, dialogue |                                     |

Here, we only list a small part of the  instruction tuning dataset list, to find more datasets, please check out the following links:
[jianzhnie/awesome-instruction-datasets](https://github.com/jianzhnie/awesome-instruction-datasets): A collection of open-source dataset to train instruction-following LLMs (ChatGPT,LLaMA,Alpaca).

### RLHF Datasets

Instruction Tuning / Reinforcement Learning from Human Feedback (RLHF) Dataset is a key component of instruction-following LLMs such as ChatGPT. Follwing is a comprehensive list of datasets used for instruction tuning in various LLMs, making it easier for researchers and developers to access and utilize these resources. 

|                           Project                            |              Org              | Nums   |  Lang   | Summary                                                      |
| :----------------------------------------------------------: | :---------------------------: | ------ | :-----: | ------------------------------------------------------------ |
| [webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) |            Openai             | 19,578 | English | In the [WebGPT paper](https://arxiv.org/abs/2112.09332), the authors trained a reward model from human feedback. They used the reward model to train a long form question answering model to align with human preferences. This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total. |
|    [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)    |          stanfordnlp          | 349 K  | English | SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)). |
| [rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets) |           yitingxie           | 76.3 k | English |                                                              |
| [Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) |            Dahoas             | 112 k  | English | Anthropic's HH dataset reformatted into prompt, chosen, rejected samples. |
| [Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) |            Dahoas             |        | English |                                                              |
| [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) |            Dahoas             | 76.3k  | English | Split of [hh-static](https://huggingface.co/datasets/Dahoas/static-hh) used for training reward models after supervised fine-tuning. |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) |           Anthropic           | 22k    | English | This RLHF dataset is an iterated 'online' dataset that includes data from 52B language models. It contains 22k helpfulness comparisons and no red-teaming data. |
| [Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | Instruction-Tuning-with-GPT-4 | 52k    | English | Ranked responses (Note: Data is evaluated by `GPT-4` model NOT human) of Alpaca prompts from three models (GPT-4, GPT-3.5 and OPT-IML) by asking GPT-4 to rate the quality. Author believes "GPT-4 is capable of identifying and fixing its own mistakes, and accurately judging the quality of responses" |
| [thu-coai/Safety-Prompts](https://github.com/thu-coai/Safety-Prompts) |           thu-coai            | 100k   | Chinese | 中文安全prompts，用于评测和提升大模型的安全性，将模型的输出与人类的价值观对齐。 |
| [Chatgpt-Comparison-Detection project](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) |                               |        |         |                                                              |

To find more datasets, please check out the following links:
[jianzhnie/awesome-instruction-datasets](https://github.com/jianzhnie/awesome-instruction-datasets): A collection of open-source dataset to train instruction-following LLMs (ChatGPT,LLaMA,Alpaca).

### Data Preprocessing

We has developed a data preprocessing code that offers a unified interface for various large language models. This code can be used to preprocess data for a variety of purposes, such as Instruct Tuning and RLHF modeling tasks. If you're interested in learning more, please check out the following links to our prompt dataset and data utilities:

- [prompt_dataset.py](https://github.com/jianzhnie/open-chatgpt/blob/main/chatgpt/dataset/prompt_dataset.py)
- [data_utils.py](https://github.com/jianzhnie/open-chatgpt/blob/main/chatgpt/dataset/data_utils.py)

### Data Fomatting

In our collection, all data has been formatted using the same templates. Each sample follows the following structure:

```
[
{"instruction": instruction string,
"input": input string, # (may be empty)
"output": output string}
]
```

## Install

```bash
git clone https://github.com/jianzhnie/open-chatgpt.git
pip install -r requirements.txt
```

**PEFT**

- If you would like to use LORA along with other parameter-efficient methods, please install [peft](https://github.com/huggingface/peft) as an additional dependency.

**DeepSpeed**

- If you want to  accelerate LLM training using techniques such as pipeline parallelism, gradient checkpointing, and tensor fusion. Please install [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## Instruction Fintune

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
