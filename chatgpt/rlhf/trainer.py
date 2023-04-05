from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler

from chatgpt.buffer.replay_buffer import (Experience, ExperienceMaker,
                                          ReplayBuffer)
from chatgpt.models.loss import PolicyLoss, ValueLoss
from chatgpt.rlhf.actor_critic import ActorModel, CriticModel
from chatgpt.rlhf.callbacks import Callback
from chatgpt.rlhf.reward_model import RewardModel


class Trainer(ABC):
    """Base class for rlhf trainers.

    Args:
        strategy (Strategy):the strategy to use for training
        experience_maker (ExperienceMaker): the experience maker to use for produce experience to fullfill replay buffer
        replay_buffer (ReplayBuffer): the replay buffer to use for training
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenizer (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        data_loader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """
    def __init__(
        self,
        experience_maker: ExperienceMaker,
        replay_buffer: ReplayBuffer,
        experience_batch_size: int = 8,
        max_epochs: int = 1,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        sample_replay_buffer: bool = False,
        dataloader_pin_memory: bool = True,
        callbacks: List[Callback] = [],
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.experience_maker = experience_maker
        self.replay_buffer = replay_buffer
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.sample_replay_buffer = sample_replay_buffer
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass

    def _make_experience(
            self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        return self.experience_maker.make_experience(inputs)

    def learn(self):
        for epoch in range(self.max_epochs):
            self._on_learn_epoch_start(epoch)
            for experience in tqdm(self.prompt_dataloader):
                self._on_learn_batch_start()
                metrics = self.training_step(experience)
                self._on_learn_batch_end(metrics, experience)
            self._on_learn_epoch_end(epoch)

    def fit(self,
            prompt_dataloader,
            num_episodes: Optional[int] = None) -> None:
        """Train the model.

        Args:
            prompt_dataloader (Dataloader): the dataloader to use for training
            num_epochs (int, optional): the number of epochs to train the model
        """
        self.prompt_dataloader = prompt_dataloader
        self._on_fit_start()
        if num_epochs is None:
            num_epochs = self.max_epochs
        for episode in range(num_episodes):
            for epoch in range(num_epochs):
                self._on_episode_start(epoch)
                for batch in tqdm(prompt_dataloader):
                    self._on_make_experience_start()
                    experience = self._make_experience(batch)
                    self._on_make_experience_end(experience)
                    self.replay_buffer.append(experience)
                self._on_episode_end(epoch)
            self._on_fit_end()

    # TODO(ver217): maybe simplify these code using context
    def _on_fit_start(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()

    def _on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end()

    def _on_episode_start(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(episode)

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experience)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict,
                            experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_end(metrics, experience)


class PPOTrainer(Trainer):
    def __init__(
        self,
        actor: ActorModel,
        critic: CriticModel,
        reward_model: RewardModel,
        initial_model: ActorModel,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        kl_coef: float = 0.1,
        ptx_coef: float = 0.9,
        train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.4,
        experience_batch_size: int = 8,
        max_epochs: int = 1,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        sample_replay_buffer: bool = False,
        dataloader_pin_memory: bool = True,
        callbacks: List[Callback] = [],
        **generate_kwargs,
    ) -> None:
        experience_maker = ExperienceMaker(actor, critic, reward_model,
                                           initial_model, kl_coef)
        replay_buffer = ReplayBuffer(train_batch_size, buffer_limit,
                                     buffer_cpu_offload)
        super().__init__(
            experience_maker,
            replay_buffer,
            experience_batch_size,
            max_epochs,
            tokenizer,
            sample_replay_buffer,
            dataloader_pin_memory,
            callbacks,
            **generate_kwargs,
        )

        self.actor = actor
        self.critic = critic
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def training_step(self, experience: Experience) -> Dict[str, Any]:

        self.actor.train()
        self.critic.train()

        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor(experience.sequences,
                                      num_actions,
                                      attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)

        # ptx loss
        if self.ptx_coef != 0:
            ptx = next(iter(self.pretrain_dataloader))['input_ids'].to(
                torch.cuda.current_device())
            label = next(iter(self.pretrain_dataloader))['labels'].to(
                torch.cuda.current_device())[:, 1:]
            attention_mask = next(iter(
                self.pretrain_dataloader))['attention_mask'].to(
                    torch.cuda.current_device())
            ptx_log_probs = self.actor.get_base_model()(
                ptx, attention_mask=attention_mask)['logits'][..., :-1, :]
            ptx_loss = self.ptx_loss_fn(
                ptx_log_probs.view(-1, ptx_log_probs.size(-1)), label.view(-1))
            actor_loss = ptx_loss * self.ptx_coef + actor_loss * (
                1 - self.ptx_coef)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # value loss
        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        critic_loss = self.critic_loss_fn(values,
                                          experience.values,
                                          experience.reward,
                                          action_mask=experience.action_mask)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return {'reward': experience.reward.mean().item()}
