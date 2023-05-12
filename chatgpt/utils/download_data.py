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
    repo_url = f'https://huggingface.co/datasets/{repo}'
    dest_dir = os.path.join(dir, repo)

    if os.path.exists(dest_dir):
        print(f'Repository {repo} already exists at {dest_dir}')
        return

    try:
        subprocess.run(['git', 'clone', repo_url, dest_dir], check=True)
        print(f'Successfully cloned repository {repo} into {dest_dir}')
    except subprocess.CalledProcessError as e:
        print(f'Error cloning repository {repo}: {e}')


def extract_gz_files(repo, directory):
    """
    Extract all .gz files in the specified directory and save as uncompressed files.

    Parameters:
        - repo (str): the name of the repository to look for in the specified directory
        - directory (str): the path to the directory where the .gz files are located
    """
    # Find all .gz files in the specified directory with the given repository name
    file_list = glob.glob(os.path.join(directory, repo, '*.gz'))

    # Loop through each .gz file and extract its contents to a new output file
    for input_file_path in file_list:
        output_file_path = os.path.splitext(input_file_path)[0]

        with gzip.open(input_file_path, 'rb') as input_file, \
             open(output_file_path, 'wb') as output_file:

            shutil.copyfileobj(input_file, output_file)

        print(f'Extracted {input_file_path} to {output_file_path}')


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
