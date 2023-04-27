# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import re

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Dataset, Subset


class PromptDataset(Dataset):
    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                'input_ids': self.chosen_dataset[idx]['input_ids'],
                'attention_mask': self.chosen_dataset[idx]['attention_mask'],
                'labels': self.chosen_dataset[idx]['input_ids']
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]['input_ids'], self.chosen_dataset[
                idx]['attention_mask'], self.reject_dataset[idx][
                    'input_ids'], self.reject_dataset[idx]['attention_mask']
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]['input_ids'], self.prompt_dataset[
                idx]['attention_mask'], self.pad_token_id


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name.replace('/', '_')
        self.output_path = output_path
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
        output_path,
        seed,
        dataset_name='Dahoas/rm-static',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='Dahoas/full-hh-rlhf',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='Dahoas/synthetic-instruct-gptj-pairwise',
    ):
        super().__init__(output_path, seed, dataset_name)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
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
        output_path,
        seed,
        dataset_name='yitingxie/rlhf-reward-datasets',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='openai/webgpt_comparisons',
    ):
        super().__init__(output_path, seed, dataset_name)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
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
        output_path,
        seed,
        dataset_name='stanfordnlp/SHP',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='wangrui6/Zhihu-KOL',
    ):
        super().__init__(output_path, seed, dataset_name)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
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
        output_path,
        seed,
        dataset_name='Cohere/miracl-zh-queries-22-12',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='Hello-SimpleAI/HC3-Chinese',
    ):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = 'Hello-SimpleAI/HC3-Chinese'
        self.dataset_name_clean = 'Hello_SimpleAI_HC3_Chinese'

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
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
        output_path,
        seed,
        dataset_name='mkqa-Chinese',
    ):
        super().__init__(output_path, seed, dataset_name)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
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
        output_path,
        seed,
        dataset_name='mkqa-Japanese',
    ):
        super().__init__(output_path, seed, dataset_name)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets['train']
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, 'train_eval', '9,1', 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
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
        output_path,
        seed,
        dataset_name='Cohere/miracl-ja-queries-22-12',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='lmqg/qg_jaquad',
    ):
        super().__init__(output_path, seed, dataset_name)

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
        output_path,
        seed,
        dataset_name='lmqg/qag_jaquad',
    ):
        super().__init__(output_path, seed, dataset_name)

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
