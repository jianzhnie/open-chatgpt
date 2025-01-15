<div align="center">
  <img src="assets/logo.png" width="800"/>
<div>&nbsp;</div>
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

- \[2023/04\] We released **RLHF(Reinforcement Learning with Human Feedback)  Pipeline** .
- \[2023/03\] We released the code **OpenChatGPT: An Open-Source libraray to train ChatBot like ChatGPT**.

## Install

```bash
git clone https://github.com/jianzhnie/open-chatgpt.git
pip install -r requirements.txt
```

**PEFT**

- If you would like to use LORA along with other parameter-efficient methods, please install [peft](https://github.com/huggingface/peft) as an additional dependency.

**DeepSpeed**

- If you want to  accelerate LLM training using techniques such as pipeline parallelism, gradient checkpointing, and tensor fusion. Please install [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License

`Open-ChatGPT` is released under the Apache 2.0 license.

## Acknowledgements

We appreciate the work by many open-source contributors, especially:

- [Hugging Face](https://huggingface.co/)
- [Vicuna](https://github.com/lm-sys/FastChat/)

## Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{open-chatgpt,
  author = {jianzhnie},
  title = {Open-ChatGPT, An open-source implementation of ChatGPT},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jianzhnie/open-chatgpt}},
}
```
