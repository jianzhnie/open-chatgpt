import argparse
import random
import sys

from torch.utils.data import DataLoader

sys.path.append('../../')
from transformers import AutoTokenizer

from chatgpt.dataset.data_utils import MiniDataset
from chatgpt.dataset.prompt_dataset import PromptDataset
from chatgpt.rlhf.ppo_trainer_deepspeed import PPOTrainer


def main():
    pretrained = 'facebook/opt-125m'
    data_path = 'CarperAI/openai_summarize_tldr'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    prompt_dataset = PromptDataset(data_path=data_path,
                                   tokenizer=tokenizer,
                                   split='valid',
                                   max_length=512)
    print(prompt_dataset)
    prompt_dataloader = DataLoader(prompt_dataset, shuffle=True, batch_size=8)
    print(prompt_dataloader)
    config = {
        'pretrained': 'facebook/opt-125m',
        'num_train_epochs': 10,
        'ppo_epoch': 10,
        'replay_buffer_size': 5000,
        'per_device_mini_train_batch_size': 4,
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'batch_size': 32,
        'total_steps': 1000000,
        'train_log_interval': 5,  # log every 10 episode
        'test_log_interval': 20,  # log every 100 epidode
        'log_dir': 'work_dirs/',
        'logger': 'wandb'
    }

    args = argparse.Namespace(**config)

    exp_dataset = MiniDataset(args.replay_buffer_size,
                              args.per_device_mini_train_batch_size)
    ppo_trainer = PPOTrainer(pretrained=args.pretrained)
    for epoch in range(10):
        print({epoch + 1} / {args.num_train_epochs})
        for step, batch_prompt in enumerate(prompt_dataloader):
            out = ppo_trainer.generate_experience(batch_prompt)
            print(out)
            exp_dataset = exp_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                critic_loss, actor_loss = 0, 0
                average_reward = 0

                for ppo_ep in range(args.ppo_epoch):
                    for i, exp_data in enumerate(exp_dataset):
                        actor_loss, critic_loss = ppo_trainer.train_rlhf(
                            exp_data)
                        critic_loss += actor_loss.item()
                        actor_loss += critic_loss.item()
                        average_reward += exp_data['rewards'].mean()
                        inner_iter += 1
                    random.shuffle(exp_dataset)

                print(f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}| \
                        act_loss: {actor_loss/inner_iter} |cri_loss: {critic_loss/inner_iter}'
                      )
                print(f'average reward score: {average_reward/inner_iter}')


if __name__ == '__main__':
    main()
