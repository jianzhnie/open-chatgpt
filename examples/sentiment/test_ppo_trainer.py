import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

sys.path.append('../../')
from chatgpt.rlhf.actor_critic import ActorCritic, ActorModel, CriticModel
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
    ds = ds.select([0, 10, 20, 30, 40, 50])
    ds = ds.filter(lambda x: len(x['review']) > 100, batched=False)
    return ds


def main():
    model_name = 'facebook/opt-125m'
    actor_model = ActorModel(model_name)
    critic_model = CriticModel(pretrained=model_name)
    actor_critic = ActorCritic(actor_model, critic_model, debug=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_size = LengthSampler(3, 12)

    ds = build_dataset()
    ds.set_format(type='torch')

    def tokenize(sample):
        sample = tokenizer(sample['review'][:input_size()])
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # We retrieve the dataloader by calling the `build_dataset` function.
    ds = build_dataset(dataset_name='imdb')
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type='torch', device=device)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    dataloader = DataLoader(ds, batch_size=4, collate_fn=collator)

    for idx, batch in enumerate(dataloader):
        ac = actor_model.generate(batch['input_ids'], batch['attention_mask'])
        print(ac)
        exit()


if __name__ == '__main__':
    main()
