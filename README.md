<div align="center">
  <img src="assets/logo.png" width="800"/>
<div>&nbsp;</div>

</div>

<div align="center">

[中文](README_zh.md) | English
</div>


## RLHF 
Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the GPT architecture.

**ChatGPT** is a conversational AI model based on the GPT-3.5 (Generative Pre-trained Transformer 3.5) architecture and is the brother model of InstructGPT. Although ChatGPT is not open source, we can see its technical framework on OpenAI's [blog](https://openai.com/blog/chatgpt).

ChatGPT continues the technical path of [InstructGPT/GPT3.5](https://arxiv.org/abs/2203.02155) and adds RLHF (Reinforcement Learning from Human Feedback) which enhances the adjustment of the model output by humans and sorts the results with greater understanding.

Reinforcement learning from human feedback (RLHF) is a challenging concept as it involves multiple model training processes and different deployment stages. We break down the training process into three core steps:

### Step 1: Train Supervised Fine-Tuning (SFT) Policy Model

GPT 3.5 itself has difficulty in understanding the different intentions implied in various types of human instructions, and it is also difficult to judge whether the generated content is of high quality. To make [GPT 3.5](https://arxiv.org/abs/2203.02155) initially understand the intent of instructions, high-quality answers are given by human annotators for randomly selected questions in the dataset, and the GPT-3.5 model is fine-tuned with these manually labeled data to obtain the SFT model (Supervised Fine-Tuning).

The SFT model at this point is already better than GPT-3 in following instructions/dialogues, but may not necessarily align with human preferences.

### Step 2: Train Reward Model (RM)

The main objective of this stage is to train a reward model by manually labeled training data (about 33K data). Questions are randomly selected from the dataset, and multiple different answers are generated for each question using the model generated in the first stage. Human annotators consider these results comprehensively and provide a ranking order. This process is similar to a coach or teacher's guidance.

Next, use this ranking result data to train the reward model. For multiple ranking results, pairwise combinations form multiple training data pairs. The RM model accepts an input and provides a score that evaluates the quality of the answer. Thus, for a pair of training data, the parameters are adjusted so that the score for a high-quality answer is higher than that for a low-quality answer.

### Step 3: Optimize the Policy Using Reinforcement Learning

Finally, once we have the trained SFT model and reward model (RM), we can use reinforcement learning (RL) to fine-tune the SFT model based on feedback using RM. This step keeps our SFT model aligned with human preferences.

This stage uses the reward model trained in the second stage and updates the pre-trained model parameters based on the reward score. Questions are randomly selected from the dataset, and the PPO model is used to generate answers, and the RM model trained in the previous stage is used to provide quality scores. The reward scores are passed in sequence, resulting in a policy gradient, and the PPO model parameters are updated through reinforcement learning.

If we repeatedly go through the second and third stages,

</div>
<img src="./assets/ChatGPT_Diagram.svg" width="600px"></img>

*<a href="https://openai.com/blog/chatgpt/">official chatgpt blogpost</a>*
</div>