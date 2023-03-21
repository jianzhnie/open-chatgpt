
## [WebGPT Comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)


### 数据集描述
在 [WebGPT](https://arxiv.org/abs/2112.09332) 论文中，作者根据人类反馈训练了一个奖励模型。他们使用奖励模型来训练长格式问答模型以符合人类偏好。这是在 WebGPT 项目结束时标记为适合奖励建模的所有比较的数据集。总共有 19,578 次比较。

数据集中的每个示例都包含一对问题的模型答案，以及相关的元数据。每个答案都有一个来自人类的偏好分数，可用于确定两个答案中哪个更好。总体而言，示例具有以下字段：

- question：问题的文本，以及从中获取问题的数据集的名称和唯一 ID。
- quotes_0：模型在浏览时找到的摘录answer_0，以及找到摘录的页面的标题，由页面的 HTML 标题和域名构成。
- answer_0: 模型使用 quotes_0 组成的最终答案。
- score_0：认为answer_0 优于 answer_1 的偏好强度，  为从 −1 到 1 的数字。它与 score_1总和为 0 ，当且仅当其分数为正时，答案才是首选。对于奖励建模，我们将 0 分视为软 50% 标签，将所有其他分数视为硬标签（仅使用它们的符号）。
- quotes_1: quotes_0 的对应。
- answer_1: answer_0 的对应。
- score_1:  score_0 的对应。

## [summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)


### 数据集描述
在[Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) 论文中，奖励模型是根据人类反馈训练的。然后使用奖励模型来训练摘要模型以符合人类偏好。这是为奖励建模而发布的人类反馈数据集。这个数据集有两个部分：comparisons和axis。在这一comparisons部分中，人类注释者被要求从两个摘要中选择最好的。在这一axis部分中，人工注释者根据李克特量表对摘要的质量进行了评分。该comparisons部分只有训练和验证拆分，该axis部分只有测试和验证拆分。

论文中用于训练奖励模型的摘要来自 TL;DR 数据集。其他验证和测试数据来自 TL;DR 数据集、CNN 文章和每日邮报文章。

