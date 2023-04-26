## Learning to summarize from Human Feedback using `open-chatgpt`

This example shows how to train a summarization model using human feedback
following the fine-tuning procedures described in Stiennon et al.'s, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)".


### Training Process

We leave the following for a quick overview of the fine-tuning process and what scripts to run.


1. Train SFT:

    ```bash
    python train_fintune_summarize.py
    ```

2. Train Reward Model:
    ```bash
    python train_reward_model.py
    ```
    Download reward model checkpoint:
    ```bash

3. PPO training:
    ```bash
    python train_ppo_rlhf.py
    ```


### Results

The following tables display ROUGE and reward scores on the test set of the TL;DR dataset between SFT and PPO models.

1. SFT vs PPO

    __ROUGE scores__

    | Model | Rouge-1 | Rouge-2 | Rouge-L | Average |
    | ----- | ------- | ------- | ------- | ------- |
    | SFT   | 0.334   | 0.125   | 0.261   | 0.240   |
    | PPO   | 0.323   | 0.109   | 0.238   | 0.223   |

    __Reward scores__

    | Model | Average Reward | Reward $\Delta$ |
    | ----- | -------------- | --------------- |
    | SFT   | 2.729          | -0.181          |
    | PPO   | 3.291          | +0.411          |


## References

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)", Neural Information Processing Systems, 2020.
