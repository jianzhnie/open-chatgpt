import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append("../../")
from chatgpt.dataset.prompt_dataset import PromptDataset
from chatgpt.rlhf.actor_critic import ActorModel, CriticModel
from chatgpt.rlhf.reward_model import RewardModel
from chatgpt.rlhf.trainer import PPOTrainer
from chatgpt.rlhf.callbacks import Callback


def main(args):
    pretrained = 'facebook/opt-125m'
    data_path = 'CarperAI/openai_summarize_tldr'
    initial_model = ActorModel(pretrained, debug=True)
    reward_model = RewardModel(model='opt', pretrained=pretrained)
    actor_model = ActorModel(pretrained=pretrained, debug=True)
    critic_model = CriticModel(model='opt', pretrained=pretrained, debug=True)

    actor_optim = Adam(actor_model.parameters(), lr=1e-7)
    critic_optim = Adam(critic_model.parameters(), lr=1e-7)

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    prompt_dataset = PromptDataset(data_path=data_path,
                                   tokenizer=tokenizer,
                                   split='valid',
                                   max_length=512)
    print(prompt_dataset)
    prompt_dataloader = DataLoader(prompt_dataset, shuffle=True, batch_size=8)
    print(prompt_dataloader)

    callbacks = [Callback()]
    # configure trainer
    trainer = PPOTrainer(
        actor=actor_model,
        critic=critic_model,
        reward_model=reward_model,
        initial_model=initial_model,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        kl_coef=args.kl_coef,
        ptx_coef=args.ptx_coef,
        train_batch_size=args.train_batch_size,
        buffer_limit=args.buffer_limit,
        experience_batch_size=args.experience_batch_size,
        max_epochs=args.max_epochs,
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        callbacks=callbacks,
    )

    trainer.train(prompt_dataloader=prompt_dataloader,
                  num_episodes=args.num_episodes)


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
