import glob
import gzip
import os
import shutil
import subprocess

data_path = [
    'Dahoas/rm-static',
    'Dahoas/full-hh-rlhf',
    'Dahoas/synthetic-instruct-gptj-pairwise',
    'yitingxie/rlhf-reward-datasets',
    'openai/webgpt_comparisons',
    'stanfordnlp/SHP',
    'wangrui6/Zhihu-KOL',
    'Cohere/miracl-zh-queries-22-12',
    'Hello-SimpleAI/HC3-Chinese',
    'mkqa-Chinese',
    'mkqa-Japanese',
    'Cohere/miracl-ja-queries-22-12',
    'lmqg/qg_jaquad',
    'lmqg/qag_jaquad',
    'lvwerra/stack-exchange-paired',
    'Anthropic/hh-rlhf',
    'databricks/databricks-dolly-15k',
    'mosaicml/dolly_hhrlhf',
    'JosephusCheung/GuanacoDataset',
    'YeungNLP/firefly-train-1.1M',
    'laion/OIG',
    'OpenAssistant/oasst1',
    'BelleGroup/train_1M_CN',
    'BelleGroup/train_0.5M_CN',
    'tatsu-lab/alpaca',
    'yahma/alpaca-cleaned',
    'QingyiSi/Alpaca-CoT',
    'fnlp/moss-002-sft-data',
    'nomic-ai/gpt4all-j-prompt-generations',
]


def clone_repo(repo, dir):
    print(
        f'git clone https://huggingface.co/datasets/{repo} into {dir}/{repo}')
    path = os.path.join(dir, repo)
    if not os.path.exists(path):
        os.makedirs(path)
    process = subprocess.run(
        f'git clone https://huggingface.co/datasets/{repo} {dir}/{repo}/',
        shell=True,
        check=True)
    print(process.stdout)


if __name__ == '__main__':
    process = subprocess.run(
        'git lfs env | grep -q \'git config filter.lfs.smudge = "git-lfs smudge -- %f"\'',
        shell=True)
    if process.returncode != 0:
        print(
            'error: git lfs not installed. please install git-lfs and run `git lfs install`'
        )

    dir = '/home/robin/work_dir/llm/open-chatgpt/prompt_data'
    for repo in data_path:
        clone_repo(repo, dir)
