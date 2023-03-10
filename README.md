# open-chatgpt
The open source implementation of chatgpt


## Use an Existing Dataset
Alternatively, training can be bootstrapped using a pre-existing dataset available on HuggingFace.  High quality candidates are namely the Anthropic HH RLHF and the Stanford Human Preference datasets.


[Anthropic HH RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
This dataset consists of structured question/response pairs with a LLM chatbot that include chosen and rejected responses.

[Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)
This dataset is curated from selected "ask" subreddits and contains questions spanning a wide array of question/answer pairs based on the most upvoted responses.  Unlike HH RLHF, this dataset is not intended to reduce harmfulness by selecting the ideal response by a chatbot but instead weights the most helpful human responses.

[rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
It is a dataset of chosen & rejected response of the same prompt.
