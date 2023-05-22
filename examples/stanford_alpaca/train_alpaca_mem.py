# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from chatgpt.models.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn, )

replace_llama_attn_with_flash_attn()

from .train_alpaca import train

if __name__ == '__main__':
    train()