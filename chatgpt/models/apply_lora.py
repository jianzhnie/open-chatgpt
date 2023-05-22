"""
Apply the LoRA weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_lora --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B

Dependency:
pip3 install git+https://github.com/huggingface/peft.git@2822398fbe896f25d4dac5e468624dc5fd65a51b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


from typing import Tuple


def apply_lora(base_model_path: str, lora_path: str, target_model_path: str=None, save_target_model: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    # Load the LoRA adapter
    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    # Apply the LoRA adapter and save the target model if required
    model = lora_model.merge_and_unload()
    if save_target_model and target_model_path is not None:
        print("Applying the LoRA")
        model = lora_model.merge_and_unload()
        print(f"Saving the target model to {target_model_path}")
        model.save_pretrained(target_model_path)
        base_tokenizer.save_pretrained(target_model_path)

    return model, base_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--save-target-model", type=bool,default=False)

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path, args.save_target_model)
