

## Collection of prompt datasets


I have collected a few datasets of prompts to train chatllm model. Follwing are the datasets:


|              dataset name               |               Dataset Class                |                                  Links                                  | Description |
| :-------------------------------------: | :----------------------------------------: | :---------------------------------------------------------------------: | :---------: |
|      lvwerra/stack-exchange-paired      |            StackExchangeParied             |      https://huggingface.co/datasets/lvwerra/stack-exchange-paired      |             |
|            Anthropic/hh-rlhf            |              AnthropicHHRLHF               |            https://huggingface.co/datasets/anthropic/hh-rlhf            |             |
|     databricks/databricks-dolly-15k     |             DatabricksDolly15k             |     https://huggingface.co/datasets/databricks/databricks-dolly-15k     |             |
|          mosaicml/dolly_hhrlhf          |            MosaicMLDollyHhrlhf             |          https://huggingface.co/datasets/mosaicml/dolly_hhrlhf          |             |
|      JosephusCheung/GuanacoDataset      |               GuanacoDataset               |      https://huggingface.co/datasets/josephuscheung/guanacodataset      |             |
|       YeungNLP/firefly-train-1.1M       |              YeungNLP_Firefly              |       https://huggingface.co/datasets/yeungnlp/firefly-train-1.1M       |             |
| instinwild_ch | InstructWildDataset | https://github.com/XueFuzhao/InstructionWild/tree/main/data | |
| instinwild_en | InstructWildDataset | https://github.com/XueFuzhao/InstructionWild/tree/main/data | |
|               llama_data      |              HuatuoMedDataset              |   https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/tree/main/data    |             |
| live_cancer | HuatuoMedDataset |  | |
|                laion/OIG                |                  LaionOIG                  |                https://huggingface.co/datasets/laion/oig                |             |
|          OpenAssistant/oasst1           |            OpenAssistantOasst1             |          https://huggingface.co/datasets/openassistant/oasst1           | 8w |
|         BelleGroup/train_1M_CN          |            BelleGroupTrain1MCN             |         https://huggingface.co/datasets/bellegroup/train_1M_CN          | 100w |
|        BelleGroup/train_0.5M_CN         |            BelleGroupTrain05MCN            |        https://huggingface.co/datasets/bellegroup/train_0.5M_CN         | 50w |
|            tatsu-lab/alpaca             |                   AlpacaDataset                   |            https://huggingface.co/datasets/tatsu-lab/alpaca             | 52k |
|          yahma/alpaca-cleaned           |               AlpacaCleaned                |          https://huggingface.co/datasets/yahma/alpaca-cleaned           | 52k |
|           QingyiSi/Alpaca-CoT           |                 AlpacaCoT                  |           https://huggingface.co/datasets/qingyisi/alpaca-cot           |             |
| trans_chinese_alpaca_data | AlpacaChinese | https://github.com/LC1332/Luotuo-Chinese-LLM/tree/main/data | |
| trans_chinese_alpaca_data | AlpacaChinese | https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/data | |
| fnlp/moss-002-sft-data | FudanMossDataset | https://huggingface.co/datasets/fnlp/moss-002-sft-data | |
| nomic-ai/gpt4all-j-prompt-generations | Gpt4allPromptGeneration | https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations | |
|            Dahoas/rm-static             |       RmStaDahoasRmstaticDatasettic        |            https://huggingface.co/datasets/dahoas/rm-static             |       8w      |
|           Dahoas/full-hh-rlhf           |          DahoasFullhhrlhfDataset           |           https://huggingface.co/datasets/dahoas/full-hh-rlhf           |         12w    |
| Dahoas/synthetic-instruct-gptj-pairwise | DahoasSyntheticinstructgptjpairwiseDataset | https://huggingface.co/datasets/dahoas/synthetic-instruct-gptj-pairwise |        3w     |
|     yitingxie/rlhf-reward-datasets      |     YitingxieRlhfrewarddatasetsDataset     |     https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets      |       8w      |
|        openai/webgpt_comparisons        |       OpenaiWebgptcomparisonsDataset       |        https://huggingface.co/datasets/openai/webgpt_comparisons        |         2w   |
|             stanfordnlp/SHP             |           StanfordnlpSHPDataset            |             https://huggingface.co/datasets/stanfordnlp/SHP             |       5w      |
|           wangrui6/Zhihu-KOL            |          Wangrui6ZhihuKOLDataset           |           https://huggingface.co/datasets/wangrui6/Zhihu-KOL            |        100w     |
|     Cohere/miracl-zh-queries-22-12      |      CohereMiraclzhqueries2212Dataset      |     https://huggingface.co/datasets/cohere/miracl-zh-queries-22-12      |        1w     |
|       Hello-SimpleAI/HC3-Chinese        |       HelloSimpleAIHC3ChineseDataset       |       https://huggingface.co/datasets/hello-simpleai/HC3-Chinese        |             |
|              mkqa-Chinese               |             MkqaChineseDataset             |              https://huggingface.co/datasets/mkqa/Chinese               |             |
|              mkqa-Japanese              |            MkqaJapaneseDataset             |              https://huggingface.co/datasets/mkqa/Japanese              |             |
|     Cohere/miracl-ja-queries-22-12      |      CohereMiracljaqueries2212Dataset      |     https://huggingface.co/datasets/cohere/miracl-ja-queries-22-12      |        1w     |
|             lmqg/qg_jaquad              |            LmqgQgJaquadDataset             |             https://huggingface.co/datasets/lmqg/qg_jaquad              |        3w     |
|             lmqg/qag_jaquad             |            LmqgQagJaquadDataset            |             https://huggingface.co/datasets/lmqg/qag_jaquad             |        1w     |



## Using for Training SFT model


```python
HuggingFaceDataClass: Dict[str, Type] = {
    'Dahoas/rm-static': DahoasRmstaticDataset,
    'Dahoas/full-hh-rlhf': DahoasFullhhrlhfDataset,
    'Dahoas/synthetic-instruct-gptj-pairwise':
    DahoasSyntheticinstructgptjpairwiseDataset,
    'yitingxie/rlhf-reward-datasets': YitingxieRlhfrewarddatasetsDataset,
    'openai/webgpt_comparisons': OpenaiWebgptcomparisonsDataset,
    'stanfordnlp/SHP': StanfordnlpSHPDataset,
    'wangrui6/Zhihu-KOL': Wangrui6ZhihuKOLDataset,
    'Cohere/miracl-zh-queries-22-12': CohereMiraclzhqueries2212Dataset,
    'Hello-SimpleAI/HC3-Chinese': HelloSimpleAIHC3ChineseDataset,
    'mkqa-Chinese': MkqaChineseDataset,
    'mkqa-Japanese': MkqaJapaneseDataset,
    'Cohere/miracl-ja-queries-22-12': CohereMiracljaqueries2212Dataset,
    'lmqg/qg_jaquad': LmqgQgjaquadDataset,
    'lmqg/qag_jaquad': LmqgQagjaquadDataset,
    'Anthropic/hh-rlhf': AnthropicHHRLHF,
    'databricks/databricks-dolly-15k': DatabricksDolly15k,
    'mosaicml/dolly_hhrlhf': MosaicmlDollyHHRLHF,
    'JosephusCheung/GuanacoDataset': GuanacoDataset,
    'YeungNLP/firefly-train-1.1M': YeungNLPFirefly,
    'OpenAssistant/oasst1': OpenAssistantOasst1,
    'tatsu-lab/alpaca': AlpacaDataset,
    'yahma/alpaca-cleaned': AlpacaDataCleaned,
    'QingyiSi/Alpaca-CoT': AlpacaCoT,
    'fnlp/moss-002-sft-data': FudanMossDataset,
    'nomic-ai/gpt4all-j-prompt-generations': Gpt4allPromptGeneration,
```

本地文件

```python
LocalDataClass：Dict[str, Type]  = {
    'stack-exchange-paired': StackExchangeParied,
    'OIG': LaionOIG,
    'train_1M_CN': BelleGroupTrain1MCN,
    'train_0.5M_CN': BelleGroupTrain05MCN,
    'llama_med': HuatuoMedDataset,
    'liver_cancer': HuatuoMedDataset,
    'instinwild_en': InstructWildDataset,
    'instinwild_ch': InstructWildDataset,
    'alpaca_data_zh_51k': AlpacaChinese,
    'trans_chinese_alpaca_data': AlpacaChinese,
}
```



## Reference

- https://github.com/yaodongC/awesome-instruction-dataset
- https://github.com/zhilizju/Awesome-instruction-tuning
- https://github.com/zhengzangw/awesome-instruction-tunning-datasets
- https://github.com/andy-yangz/Awesome-Chinese-Instruction-Datasets
