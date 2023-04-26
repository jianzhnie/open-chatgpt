import sys

sys.path.append('../../')
import torch
from transformers import AutoTokenizer

from chatgpt.dataset.prompt_dataset import TokenizedPromptDataset
from chatgpt.rlhf.trainer import PPOTrainer


def main():
    pretrained = 'facebook/opt-125m'
    data_path = 'CarperAI/openai_summarize_tldr'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    prompt_dataset = TokenizedPromptDataset(data_path=data_path,
                                            tokenizer=tokenizer,
                                            split='valid',
                                            max_length=256)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: ', device)
    ppo_trainer = PPOTrainer(pretrained=pretrained,
                             prompt_dataset=prompt_dataset,
                             device=device)
    ppo_trainer.train()


if __name__ == '__main__':
    main()
