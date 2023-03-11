# Open ChatGPT

<div align="center">
  <img src="assets/logo.png" width="600"/>
<div>&nbsp;</div>

[🤖 强化学习](https://jianzhnie.github.io/machine-learning-wiki/#/deep-rl/) |
[🙆‍♀️ RLHF](https://jianzhnie.github.io/machine-learning-wiki/#/deep-rl/papers/RLHF) |
[🙆‍♀️ AI tools](https://jianzhnie.github.io/machine-learning-wiki/#/ai-general/ai-tools) |

</div>

<div align="center">

[English](README.md) | 简体中文
</div>


OpenAI 推出的 ChatGPT 对话模型掀起了新的 AI 热潮，它面对多种多样的问题对答如流，似乎已经打破了机器和人的边界。这一工作的背后是大型语言模型 (Large Language Model，LLM) 生成领域的新训练范式：RLHF (Reinforcement Learning from Human Feedback) ，即以强化学习方式依据人类反馈优化语言模型。

## 人类反馈强化学习(RLHF)

**ChatGPT** 是基于GPT-3.5（Generative Pre-trained Transformer 3.5）架构开发的对话AI模型，是InstructGPT 的兄弟模型。ChatGPT 虽然没有开源，但是我们可以在OpenAI  的官方[博客](https://openai.com/blog/chatgpt) 看到其技术框架。

ChatGPT  延续了[InstructGPT/GPT3.5](https://arxiv.org/abs/2203.02155) 的技术路线，加入了被称为RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）。这一训练范式增强了人类对模型输出结果的调节，并且对结果进行了更具理解性的排序。

从人类反馈中进行强化学习（RLHF）是一个具有挑战性的概念，因为它涉及多模型训练过程和不同的部署阶段。我们将训练过程分解为三个核心步骤：

### 第一阶段：训练监督策略模型(SFT)

GPT 3.5本身很难理解人类不同类型指令中蕴含的不同意图，也很难判断生成内容是否是高质量的结果。为了让[GPT 3.5](https://arxiv.org/abs/2203.02155)初步具备理解指令的意图，首先会在数据集中随机抽取问题，由人类标注人员，给出高质量答案，然后用这些人工标注好的数据来微调 GPT-3.5模型（获得SFT模型, Supervised Fine-Tuning）。

此时的SFT模型在遵循指令/对话方面已经优于 GPT-3，但不一定符合人类偏好。

### 第二阶段：训练奖励模型（Reward Model，RM）

这个阶段的主要是通过人工标注训练数据（约33K个数据），来训练回报模型。在数据集中随机抽取问题，使用第一阶段生成的模型，对于每个问题，生成多个不同的回答。人类标注者对这些结果综合考虑给出排名顺序。这一过程类似于教练或老师辅导。

接下来，使用这个排序结果数据来训练奖励模型。对多个排序结果，两两组合，形成多个训练数据对。RM模型接受一个输入，给出评价回答质量的分数。这样，对于一对训练数据，调节参数使得高质量回答的打分比低质量的打分要高。

### 第三阶段：采用强化学习来优化策略（RLHF）

最后，一旦我们有了训练好的SFT 模型和奖励模型(RM)，现在可以通过强化学习(RL) 使用 RM来根据反馈微调 SFT 模型。此步骤使我们的 SFT 模型与人类偏好保持一致。

这一阶段利用第二阶段训练好的奖励模型，靠奖励打分来更新预训练模型参数。在数据集中随机抽取问题，使用强化学习中的近端策略优化（Proximal Policy Optimization，PPO）算法引入奖励信号，指导模型生成回答，并用上一阶段训练好的RM模型给出质量分数。把回报分数依次传递，由此产生策略梯度，通过强化学习的方式以更新PPO模型参数。

如果我们不断重复第二和第三阶段，通过**迭代**，会训练出更高质量的ChatGPT模型。

为了更好地理解此过程，请查看下图概述的三个阶段。

## 数据集

另外，可以使用HuggingFace上提供的现有数据集来引导训练。其中高质量的候选数据集包括Anthropic HH RLHF和Stanford Human Preference数据集。

- [Anthropic HH RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) 

该数据集包含结构化的问题/回答对和一个带有LLM聊天机器人的选定和被拒绝的回答。

- [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP) 

该数据集是从选定的“问答”子版面中精选出来的，包含了涵盖广泛的问题/答案对，这些问题是基于最受欢迎的回答而选择的。与HH RLHF不同，这个数据集的目的不是通过选择聊天机器人的理想回答来减少有害性，而是通过加权最有帮助的人类回答来实现这一目的。

- [Reddit TL;DR dataset](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr)

TL;DR 摘要数据集是从Reddit精心挑选的帖子，包含了帖子主要内容以及人类总结的摘要。该数据集包含 129,722 个 Reddit 帖子，其中约 5% 用于拆分验证和测试。训练集中总共有 116,722 个样本，验证集中有 6,447 个样本，测试集中有 6,553 个样本。我们将使用此数据集来微调我们的模型。

- [Comparisons dataset](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons)

对于每个 Reddit 帖子, 使用预训练模型生成摘要，人工编写的摘要也被视为参考样本。每个帖子的这两个摘要被成对发送给雇佣的数据标注员，标注员根据自己的喜好选择/偏爱其中一个摘要。

该数据集由训练数据集中的 92,858 个样本和验证集中的 83,797 个样本组成。包含 Reddit 帖子和每个帖子的两个摘要， 它还具有一个选择值，指示人工标记者更喜欢两个摘要中的哪一个。



## 训练 RLHF

### Clone 代码

要开始，请首先按照下面的安装指南进行操作：

```python
git clone https://github.com/jianzhnie/open-chatgpt
pip install -r requirements.txt
```

### 监督微调 (SFT)

接下来，我们将在 `TL;DR `数据集上微调 OPT 模型以进行文本摘要。

这是相对简单的。加载数据集，对其进行 tokenize ，然后训练模型。整个 pipeline 是使用 HuggingFace 构建的。

```python
cd scripts/
python train_opt_summarize.py
```

模型使用 ROUGE 分数进行评估。验证集上的平均 ROUGE 分数选择最佳模型。该模型将用于初始化奖励模型，稍后将使用 PPO 进行微调。

### 训练奖励模型

我们的奖励模型是用收集到的人类质量判断数据集训练的。该模型将给定的帖子和候选摘要映射到奖励*r* 。

我们将从 SFT 模型初始化奖励模型，并附加一个随机初始化的线性头，在顶部输出标量值。

接下来，我们将更详细地研究数据如何输入到模型、损失函数和奖励模型的其他问题。

```python
python train_reward_model_opt.py
```

### 使用 PPO 进行微调

```python

```