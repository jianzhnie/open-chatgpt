import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    text = 'This is a great [MASK].'

    inputs = tokenizer(text, return_tensors='pt')
    token_logits = model(**inputs).logits
    print(token_logits.shape)
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(
        inputs['input_ids'].cpu() == tokenizer.mask_token_id)[1]

    print(mask_token_index)

    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(
            f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'"
        )
