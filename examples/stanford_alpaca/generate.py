import argparse
import sys
from typing import Tuple, Union

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, LlamaTokenizer)

sys.path.append('../../')


class Prompter(object):
    def __init__(self) -> None:
        self.PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task, paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
             ),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }
        self.reponse_split = '### Response:'

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        response: Union[None, str] = None,
    ):
        prompt_input, prompt_no_input = self.PROMPT_DICT[
            'prompt_input'], self.PROMPT_DICT['prompt_no_input']
        if input is not None:
            prompt_text = prompt_input.format(instruction=instruction,
                                              input=input)
        else:
            prompt_text = prompt_no_input.format(instruction=instruction)

        if response:
            prompt_text = f'{prompt_text}{response}'
        return prompt_text

    def get_response(self, output: str) -> str:
        return output.split(self.reponse_split)[1].strip()


def apply_lora(
    base_model_path: str,
    lora_path: str,
    load_8bit: bool = False,
    target_model_path: str = None,
    save_target_model: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Applies the LoRA adapter to a base model and saves the resulting target model (optional).

    Args:
        base_model_path (str): The path to the base model to which the LoRA adapter will be applied.
        lora_path (str): The path to the LoRA adapter.
        target_model_path (str): The path where the target model will be saved (if `save_target_model=True`).
        save_target_model (bool, optional): Whether to save the target model or not. Defaults to False.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the target model and its tokenizer.

    """
    # Load the base model and tokenizer
    print(f'Loading the base model from {base_model_path}')
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True)

    # Load the tokenizer
    if base_model.config.model_type == 'llama':
        # Due to the name of Transformers' LlamaTokenizer, we have to do this
        base_tokenizer = LlamaTokenizer.from_pretrained(
            base_model_path,
            padding_side='right',
            use_fast=True,
        )
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side='right',
            use_fast=True,
        )

    # Load the LoRA adapter
    print(f'Loading the LoRA adapter from {lora_path}')
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )

    if save_target_model and target_model_path is not None:
        print(f'Saving the target model to {target_model_path}')
        model.save_pretrained(target_model_path)
        base_tokenizer.save_pretrained(target_model_path)

    return model, base_tokenizer


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        default=None,
        type=str,
        required=True,
        help=
        'Path to pre-trained model or shortcut name selected in the list: ',
    )
    parser.add_argument(
        '--lora_model_name_or_path',
        default=None,
        type=str,
        required=True,
        help=
        'Path to pre-trained model or shortcut name selected in the list: ',
    )
    parser.add_argument('--stop_token',
                        type=str,
                        default=None,
                        help='Token at which text generation is stopped')
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help=
        'temperature of 1.0 has no effect, lower tend toward greedy sampling',
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='primarily useful for CTRL model; in that case, use 1.2')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_beams', type=int, default=0)
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='Text added prior to input.')
    parser.add_argument('--padding_text',
                        type=str,
                        default='',
                        help='Deprecated, the use of `--prefix` is preferred.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for initialization')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--num_return_sequences',
                        type=int,
                        default=1,
                        help='The number of samples to generate.')
    parser.add_argument(
        '--load_8bit',
        action='store_true',
        help=
        'Whether to use load_8bit  instead of 32-bit',
    )
    args = parser.parse_args()

    args.device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    return args


def complete_prompts(model, tokenizer, generation_config, prompter,
                     prompt_text, device):
    inputs = tokenizer(prompt_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)


def main(args):
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=True,
        no_repeat_ngram_size=6,
        repetition_penalty=1.8,
        num_return_sequences=args.num_return_sequences)

    model, tokenizer = apply_lora(args.model_name_or_path,
                                  args.lora_model_name_or_path,
                                  load_8bit=args.load_8bit)
    prompter = Prompter()

    instruction_list = [
        'Tell me about alpacas.',
        'Tell me about the president of Mexico in 2019.',
        'Tell me about the king of France in 2019.',
        'List all Canadian provinces in alphabetical order.',
        'Write a Python program that prints the first 10 Fibonacci numbers.',
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        'Count up from 1 to 500.',
    ]
    # testing code for readme
    for instruction in instruction_list:
        prompt_text = prompter.generate_prompt(instruction, input=None)
        print(prompt_text)
        print(args)
        result = complete_prompts(model,
                                  tokenizer,
                                  generation_config,
                                  prompter,
                                  prompt_text,
                                  device=args.device)
        print(result)


if __name__ == '__main__':
    args = args_parser()
    main(args)
