import argparse
import sys

import torch

sys.path.append('../../')
from chatgpt.rlhf.ppo_trainer import PPOTrainer


def main(args):
    pretrained = 'facebook/opt-125m'
    data_path = 'CarperAI/openai_summarize_tldr'
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: ', device)
    # configure trainer
    trainer = PPOTrainer(
        prompt_data_path=data_path,
        pretrained_model=pretrained,
        device='cpu',
        debug=False,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path',
                        type=str,
                        default=None,
                        help='path to the prompt dataset')
    parser.add_argument('--pretrain_dataset',
                        type=str,
                        default=None,
                        help='path to the pretrained dataset')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--buffer_limit', type=int, default=32)
    parser.add_argument('--ptx_batch_size', type=int, default=1)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--ptx_coef', type=float, default=0.9)
    args = parser.parse_args()
    main(args)
