
import sys

sys.path.append('../../')
from transformers import ( AutoTokenizer
                        )

from chatgpt.dataset.data_utils import create_prompt_dataset

if __name__ == '__main__':
    train_phase = 1
    data_path = ['Dahoas/rm-static']
    data_output_path = 'work_dir/data'
    pretrained = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(pretrained, fast_tokenizer=True)
    train_dataset, eval_dataset = create_prompt_dataset(
        data_path,
        data_split='6,2,2',
        output_path=data_output_path,
        train_phase=train_phase,
        seed=42,
        tokenizer=tokenizer,
        max_seq_len=512,
        sft_only_data_path=[])
