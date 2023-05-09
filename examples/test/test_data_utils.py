import sys

sys.path.append('../../')
from transformers import AutoTokenizer

from chatgpt.dataset.data_utils import create_prompt_dataset

if __name__ == '__main__':
    train_phase = 1
    data_path = [
        # 'Dahoas/rm-static',
        # 'Dahoas/full-hh-rlhf',
        # 'Dahoas/synthetic-instruct-gptj-pairwise',
        # 'yitingxie/rlhf-reward-datasets',
        # 'openai/webgpt_comparisons',
        # 'stanfordnlp/SHP',
        # 'wangrui6/Zhihu-KOL',
        # 'Cohere/miracl-zh-queries-22-12',
        # # 'Hello-SimpleAI/HC3-Chinese',
        # # 'mkqa-Chinese',
        # # 'mkqa-Japanese',
        # 'Cohere/miracl-ja-queries-22-12',
        # 'lmqg/qg_jaquad',
        # 'lmqg/qag_jaquad',
        'lvwerra/stack-exchange-paired',
        'Anthropic/hh-rlhf',
        'databricks/databricks-dolly-15k',
        'mosaicml/dolly_hhrlhf',
        'JosephusCheung/GuanacoDataset',
        'YeungNLP/firefly-train-1.1M',
        'laion/OIG',
        'OpenAssistant/oasst1',
        # 'BelleGroup/train_1M_CN',
        # 'BelleGroup/train_0.5M_CN',
        'tatsu-lab/alpaca',
        'yahma/alpaca-cleaned',
        'QingyiSi/Alpaca-CoT',
        './prompt_data/huatuo_llama_med/llama_data.json',
    ]

    data_output_path = 'work_dir/data'
    pretrained = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(pretrained, fast_tokenizer=True)
    train_dataset, eval_dataset = create_prompt_dataset(
        dataset_names=data_path,
        train_phase=train_phase,
        test_data_ratio=0.1,
        tokenizer=tokenizer,
        output_path=data_output_path,
        max_seq_len=512)
