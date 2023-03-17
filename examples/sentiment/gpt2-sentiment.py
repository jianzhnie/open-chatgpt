import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

sys.path.append('../../')
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from chatgpt.utils.utils import LengthSampler


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(dataset_name='imdb'):
    """Build dataset for training. This builds the dataset from `load_dataset`,
    one should customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # load imdb with datasets
    ds = load_dataset(dataset_name, split='train')
    ds = ds.rename_columns({'text': 'review'})
    ds = ds.filter(lambda x: len(x['review']) > 100, batched=False)
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main():
    model_name = 'lvwerra/gpt2-imdb'
    # Now let's build the model, the reference model, and the tokenizer.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token
    input_size = LengthSampler(3, 12)

    def tokenize(sample):
        sample['input_ids'] = tokenizer.encode(sample['review'])[:input_size()]
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    # We retrieve the dataloader by calling the `build_dataset` function.
    ds = build_dataset(dataset_name='imdb')
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type='torch')

    config = PPOConfig()
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(config,
                             model,
                             ref_model,
                             tokenizer,
                             dataset=ds,
                             data_collator=collator)

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else 'cpu'
    # to avoid a `pipeline` bug
    sentiment_pipe = pipeline('sentiment-analysis',
                              model='lvwerra/distilbert-imdb',
                              device=device)

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        'min_length': -1,
        'top_k': 0.0,
        'top_p': 1.0,
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
    }
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        'return_all_scores': True,
        'function_to_apply': 'none',
        'batch_size': 16
    }

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch['input_ids']

        # Get response from gpt2
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs['max_new_tokens'] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch['response'] = [
            tokenizer.decode(r.squeeze()) for r in response_tensors
        ]

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch['query'], batch['response'])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]['score']) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == '__main__':
    main()
