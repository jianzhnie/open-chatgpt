import argparse
import sys
from typing import Union

import torch

sys.path.append('../../')
from transformers import GenerationConfig

from chatgpt.models.apply_lora import apply_lora


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
        help='Whether to use load_8bit  instead of 32-bit',
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
