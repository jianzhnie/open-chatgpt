

## Collection of prompt datasets


I have collected a few datasets of prompts to train chatllm model. Follwing are the datasets:


|       |              dataset name               |               Dataset Class                |                                  Links                                  | Description |
| :---: | :-------------------------------------: | :----------------------------------------: | :---------------------------------------------------------------------: | :---------: |
|   1   |      lvwerra/stack-exchange-paired      |            StackExchangeParied             |      https://huggingface.co/datasets/lvwerra/stack-exchange-paired      |             |
|   2   |            Anthropic/hh-rlhf            |              AnthropicHHRLHF               |            https://huggingface.co/datasets/anthropic/hh-rlhf            |             |
|   3   |     databricks/databricks-dolly-15k     |             DatabricksDolly15k             |     https://huggingface.co/datasets/databricks/databricks-dolly-15k     |             |
|   4   |          mosaicml/dolly_hhrlhf          |            MosaicMLDollyHhrlhf             |          https://huggingface.co/datasets/mosaicml/dolly_hhrlhf          |             |
|   5   |      JosephusCheung/GuanacoDataset      |               GuanacoDataset               |      https://huggingface.co/datasets/josephuscheung/guanacodataset      |             |
|   6   |       YeungNLP/firefly-train-1.1M       |              YeungNLP_Firefly              |       https://huggingface.co/datasets/yeungnlp/firefly-train-1.1M       |             |
|   7   |               Local/下载                |              HuatuoMedDataset              |   https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/tree/main/data    |             |
|   8   |                laion/OIG                |                  LaionOIG                  |                https://huggingface.co/datasets/laion/oig                |             |
|   9   |          OpenAssistant/oasst1           |            OpenAssistantOasst1             |          https://huggingface.co/datasets/openassistant/oasst1           |             |
|  10   |         BelleGroup/train_1M_CN          |            BelleGroupTrain1MCN             |         https://huggingface.co/datasets/bellegroup/train_1M_CN          |             |
|  11   |        BelleGroup/train_0.5M_CN         |           BelleGroupTrain0.5MCN            |        https://huggingface.co/datasets/bellegroup/train_0.5M_CN         |             |
|  12   |            tatsu-lab/alpaca             |                   Alpaca                   |            https://huggingface.co/datasets/tatsu-lab/alpaca             |             |
|  13   |          yahma/alpaca-cleaned           |               AlpacaCleaned                |          https://huggingface.co/datasets/yahma/alpaca-cleaned           |             |
|  14   |           QingyiSi/Alpaca-CoT           |                 AlpacaCoT                  |           https://huggingface.co/datasets/qingyisi/alpaca-cot           |             |
|  15   |            Dahoas/rm-static             |       RmStaDahoasRmstaticDatasettic        |            https://huggingface.co/datasets/dahoas/rm-static             |             |
|  16   |           Dahoas/full-hh-rlhf           |          DahoasFullhhrlhfDataset           |           https://huggingface.co/datasets/dahoas/full-hh-rlhf           |             |
|  17   | Dahoas/synthetic-instruct-gptj-pairwise | DahoasSyntheticinstructgptjpairwiseDataset | https://huggingface.co/datasets/dahoas/synthetic-instruct-gptj-pairwise |             |
|  18   |     yitingxie/rlhf-reward-datasets      |     YitingxieRlhfrewarddatasetsDataset     |     https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets      |             |
|  19   |        openai/webgpt_comparisons        |       OpenaiWebgptcomparisonsDataset       |        https://huggingface.co/datasets/openai/webgpt_comparisons        |             |
|  20   |             stanfordnlp/SHP             |           StanfordnlpSHPDataset            |             https://huggingface.co/datasets/stanfordnlp/SHP             |             |
|  21   |           wangrui6/Zhihu-KOL            |          Wangrui6ZhihuKOLDataset           |           https://huggingface.co/datasets/wangrui6/Zhihu-KOL            |             |
|  22   |     Cohere/miracl-zh-queries-22-12      |      CohereMiraclzhqueries2212Dataset      |     https://huggingface.co/datasets/cohere/miracl-zh-queries-22-12      |             |
|  23   |       Hello-SimpleAI/HC3-Chinese        |       HelloSimpleAIHC3ChineseDataset       |       https://huggingface.co/datasets/hello-simpleai/HC3-Chinese        |             |
|  24   |              mkqa-Chinese               |             MkqaChineseDataset             |              https://huggingface.co/datasets/mkqa/Chinese               |             |
|  25   |              mkqa-Japanese              |            MkqaJapaneseDataset             |              https://huggingface.co/datasets/mkqa/Japanese              |             |
|  26   |     Cohere/miracl-ja-queries-22-12      |      CohereMiracljaqueries2212Dataset      |     https://huggingface.co/datasets/cohere/miracl-ja-queries-22-12      |             |
|  27   |             lmqg/qg_jaquad              |            LmqgQgJaquadDataset             |             https://huggingface.co/datasets/lmqg/qg_jaquad              |             |
|  28   |             lmqg/qag_jaquad             |            LmqgQagJaquadDataset            |             https://huggingface.co/datasets/lmqg/qag_jaquad             |             |



## Using for Training SFT model


"""python


dataset_list = [

"lvwerra/stack-exchange-paired"

]
```


##  Using for training RM model


"""python


dataset_list = [


]
```
