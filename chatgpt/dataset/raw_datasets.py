import re
from typing import List, Tuple

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizer


def get_dataset_split_index(data_size, test_data_ratio, seed):

    index_list = list(range(data_size))
    triain_index, test_index = train_test_split(index_list,
                                                test_size=test_data_ratio,
                                                random_state=seed)
    return triain_index, test_index


class PromptDataset(Dataset):
    """
    A PyTorch dataset class that prepares prompt sentences and their corresponding chosen/rejected sentences for training.

    Args:
        prompt_dataset (List): A list of prompt sentences.
        chosen_dataset (List): A list of chosen sentences.
        reject_dataset (List): A list of rejected sentences.
        tokenizer (PreTrainedTokenizer): A pre-trained tokenizer from the Hugging Face transformers library.
        max_length (int): Maximum length of encoded sequences. Default is 512.
        train_phase (int): Phase of training data to prepare. Can be 1, 2 or 3. Default is 1.

    Returns:
        Dictionary or tuple of tensors depending on the value of train_phase.

    Examples:
        >>> prompt_dataset = ['What is your favorite color?', 'Do you like pizza?']
        >>> chosen_dataset = ['My favorite color is blue.', 'I love pizza!']
        >>> reject_dataset = ['I don\'t have a favorite color.', 'Pizza is not my thing.']
        >>> tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        >>> dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset, tokenizer, max_length=256)

    """
    def __init__(self,
                 prompt_dataset: List[str],
                 chosen_dataset: List[str],
                 reject_dataset: List[str],
                 tokenizer: PreTrainedTokenizer = None,
                 max_length: int = 512,
                 train_phase: int = 1) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.train_phase = train_phase

    def __len__(self) -> int:
        """
        Returns the length of chosen_dataset, or prompt_dataset if train_phase is 3.

        Returns:
            int: Length of dataset.
        """
        if self.train_phase == 3:
            return len(self.prompt_dataset)
        else:
            return len(self.chosen_dataset)

    def __getitem__(self, idx: int) -> dict or Tuple[torch.Tensor]:
        """
        Returns a dictionary or tuple of tensors depending on the value of train_phase.

        Args:
            idx (int): Index of dataset item to retrieve.

        Returns:
            Dictionary or tuple of tensors depending on the value of train_phase.
        """
        if self.train_phase == 1:
            raw_input = self.chosen_dataset[idx]
            encoding_input = self.tokenizer(raw_input,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length')
            # set labels equal to input_ids to enable computing loss later
            encoding_input['labels'] = encoding_input['input_ids']
            encoding_input = {
                key: torch.tensor(val)
                for key, val in encoding_input.items()
            }
            return encoding_input

        elif self.train_phase == 2:
            chosen_sentence = self.chosen_dataset[idx]
            reject_sentence = self.reject_dataset[idx]
            chosen_input = self.tokenizer(chosen_sentence,
                                          truncation=True,
                                          max_length=self.max_length,
                                          padding='max_length')
            reject_input = self.tokenizer(reject_sentence,
                                          truncation=True,
                                          max_length=self.max_length,
                                          padding='max_length')

            chosen_input = {
                key: torch.tensor(val)
                for key, val in chosen_input.items()
            }
            reject_input = {
                key: torch.tensor(val)
                for key, val in reject_input.items()
            }

            return (chosen_input['input_ids'], chosen_input['attention_mask'],
                    chosen_input['labels'], reject_input['input_ids'],
                    reject_input['attention_mask'])

        elif self.train_phase == 3:
            raw_input = self.prompt_dataset[idx]
            encoding_input = self.tokenizer(raw_input,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length')
            encoding_input = {
                key: torch.tensor(val)
                for key, val in encoding_input.items()
            }

            return (encoding_input['input_ids'],
                    encoding_input['attention_mask'], self.pad_token_id)


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self,
                 dataset_name: str = None,
                 test_data_ratio: float = 0.1,
                 seed: int = None):
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name.replace('/', '_')
        self.test_data_ratio = test_data_ratio
        self.seed = seed
        self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='Dahoas/rm-static',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='Dahoas/full-hh-rlhf',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(
            dataset_name,
            test_data_ratio,
            seed,
        )

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='Dahoas/synthetic-instruct-gptj-pairwise',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

        self.dataset = self.raw_datasets['train']
        self.train_index, self.eval_index = get_dataset_split_index(
            data_size=len(self.dataset),
            test_data_ratio=self.test_data_ratio,
            seed=self.seed,
        )

    def get_train_data(self):

        dataset = Subset(self.dataset, self.train_index)
        return dataset

    def get_eval_data(self):
        dataset = Subset(self.dataset, self.eval_index)
        return dataset

    def get_prompt(self, sample):
        return ' Human: ' + sample['prompt'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['chosen']

    def get_rejected(self, sample):
        return ' ' + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['prompt'] + ' Assistant: ' + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['prompt'] + ' Assistant: ' + sample[
            'rejected']


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='yitingxie/rlhf-reward-datasets',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt'] + 'Assistant:'

    def get_chosen(self, sample):
        return sample['chosen'].split('Assistant:')[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split('Assistant:')[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='openai/webgpt_comparisons',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

        self.dataset = self.raw_datasets['train']
        self.train_index, self.eval_index = get_dataset_split_index(
            data_size=len(self.dataset),
            test_data_ratio=self.test_data_ratio,
            seed=self.seed,
        )

    def get_train_data(self):

        dataset = Subset(dataset, self.train_index)
        return dataset

    def get_eval_data(self):

        dataset = Subset(dataset, self.eval_index)
        return dataset

    def get_prompt(self, sample):
        return ' Human: ' + sample['question']['full_text'] + ' Assistant:'

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' ' + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' ' + response

    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' Human: ' + sample['question'][
            'full_text'] + ' Assistant: ' + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' Human: ' + sample['question'][
            'full_text'] + ' Assistant: ' + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='stanfordnlp/SHP',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return ' Human: ' + sample['history'] + ' Assistant:'

    def get_chosen(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_A']
        else:
            response = sample['human_ref_B']
        return ' ' + response

    def get_rejected(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_B']
        else:
            response = sample['human_ref_A']
        return ' ' + response

    def get_prompt_and_chosen(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_A']
        else:
            response = sample['human_ref_B']
        return ' Human: ' + sample['history'] + ' Assistant: ' + response

    def get_prompt_and_rejected(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_B']
        else:
            response = sample['human_ref_A']
        return ' Human: ' + sample['history'] + ' Assistant: ' + response


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='wangrui6/Zhihu-KOL',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

        self.dataset = self.raw_datasets['train']
        self.train_index, self.eval_index = get_dataset_split_index(
            data_size=len(self.dataset),
            test_data_ratio=self.test_data_ratio,
            seed=self.seed,
        )

    def get_train_data(self):

        dataset = Subset(dataset, self.train_index)
        return dataset

    def get_eval_data(self):

        dataset = Subset(dataset, self.eval_index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return ' Human: ' + sample['INSTRUCTION'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return ' ' + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return ' Human: ' + sample[
                'INSTRUCTION'] + ' Assistant: ' + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='Cohere/miracl-zh-queries-22-12',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['dev']

    def get_prompt(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return ' ' + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='Hello-SimpleAI/HC3-Chinese',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

        self.dataset = self.raw_datasets['train']
        self.train_index, self.eval_index = get_dataset_split_index(
            data_size=len(self.dataset),
            test_data_ratio=self.test_data_ratio,
            seed=self.seed,
        )

    def get_train_data(self):

        dataset = Subset(dataset, self.train_index)
        return dataset

    def get_eval_data(self):

        dataset = Subset(dataset, self.eval_index)
        return dataset

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return ' Human: ' + sample['question'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return ' ' + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return ' Human: ' + sample['question'] + ' Assistant: ' + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='mkqa-Chinese',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

        self.dataset = self.raw_datasets['train']
        self.train_index, self.eval_index = get_dataset_split_index(
            data_size=len(self.dataset),
            test_data_ratio=self.test_data_ratio,
            seed=self.seed,
        )

    def get_train_data(self):

        dataset = Subset(dataset, self.train_index)
        return dataset

    def get_eval_data(self):

        dataset = Subset(dataset, self.eval_index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return ' Human: ' + sample['queries']['zh_cn'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return ' ' + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return ' Human: ' + sample['queries'][
                'zh_cn'] + ' Assistant: ' + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='mkqa-Japanese',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

        self.dataset = self.raw_datasets['train']
        self.train_index, self.eval_index = get_dataset_split_index(
            data_size=len(self.dataset),
            test_data_ratio=self.test_data_ratio,
            seed=self.seed,
        )

    def get_train_data(self):

        dataset = Subset(dataset, self.train_index)
        return dataset

    def get_eval_data(self):

        dataset = Subset(dataset, self.eval_index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return ' Human: ' + sample['queries']['ja'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return ' ' + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return ' Human: ' + sample['queries'][
                'ja'] + ' Assistant: ' + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='Cohere/miracl-ja-queries-22-12',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['dev']

    def get_prompt(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return ' ' + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'negative_passages'][0]['text']


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='lmqg/qg_jaquad',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['sentence']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant: ' + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):
    def __init__(
        self,
        dataset_name='lmqg/qag_jaquad',
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, test_data_ratio, seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return ' Human: ' + sample['questions'][0] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['questions'][0] + ' Assistant: ' + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None
