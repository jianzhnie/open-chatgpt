import argparse
import sys

from torch.utils.data import DataLoader

sys.path.append('../../')
from collections import deque
import torch
from transformers import AutoTokenizer

from chatgpt.buffer.replay_buffer import DsExperienceDataset, DsMemory
from chatgpt.dataset.prompt_dataset import TokenizedPromptDataset
from chatgpt.rlhf.ppo_trainer_deepspeed import PPOTrainer


def main():
    pretrained = 'facebook/opt-125m'
    data_path = 'CarperAI/openai_summarize_tldr'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    prompt_dataset = TokenizedPromptDataset(data_path=data_path,
                                            tokenizer=tokenizer,
                                            split='valid',
                                            max_length=256)
    prompt_dataloader = DataLoader(prompt_dataset, shuffle=True, batch_size=32)
    config = {
        'pretrained': 'facebook/opt-125m',
        'num_episode': 10,
        'ppo_epochs': 1,
        'gamma': 0.99,
        'batch_size': 32,
        'total_steps': 1000000,
        'train_log_interval': 5,  # log every 10 episode
        'test_log_interval': 20,  # log every 100 epidode
        'log_dir': 'work_dirs/',
        'logger': 'wandb'
    }
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: ', device)

    args = argparse.Namespace(**config)
    memories = deque([])
    ppo_trainer = PPOTrainer(pretrained=args.pretrained, device=device)
    for episode in range(args.num_episode):
        print(f"Start generating experience:, episode: {episode}")
        for step, batch_prompt in enumerate(prompt_dataloader):
            print(f"Generating experience:, episode: {episode}, step: {step}")
            out = ppo_trainer.generate_experience(batch_prompt)
            print("Putting experience into replay buffer")
            for i in range(out['prompts'].shape[0]):
                memories.append(
                    DsMemory(
                        out['prompts'][i, :].detach().cpu(),
                        out['logprobs'][i, :].detach().cpu(),
                        out['ref_logprobs'][i, :].detach().cpu(),
                        out['value'][i, :].detach().cpu(),
                        out['rewards'][i].detach().cpu(),
                        out['input_ids'][i, :].detach().cpu(),
                        out['attention_mask'][i, :].detach().cpu(),
                    ))
        print("Finished generating experience, start training")
        if memories is not None:
            inner_iter = 0
            critic_loss = 0
            actor_loss = 0

        dataset = DsExperienceDataset(memories)
        dataloader = DataLoader(dataset, batch_size=8)
        for epoch in range(args.ppo_epochs):
            print("Training RLHF")
            for i, exp_data in enumerate(dataloader):
                print(f"Training RLHF, epoch: {epoch}, step: {i}")
                actor_loss, critic_loss = ppo_trainer.train_rlhf(exp_data)
                critic_loss += actor_loss.item()
                actor_loss += critic_loss.item()
                inner_iter += 1
                print(
                    f'epoch: {epoch}|step: {i}| actor_loss: {actor_loss/inner_iter} |cri_loss: {critic_loss/inner_iter}'
                )
        print(f"Finished training RLHF, episode: {episode} ")

    print("Finished training RLHF !!!")


if __name__ == '__main__':
    main()
