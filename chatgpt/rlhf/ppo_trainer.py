from collections import deque
from typing import Deque, Optional, Tuple

import torch
import torch.optim as optim
from einops import rearrange
from torch.utils.data import DataLoader
from torchtyping import TensorType

from chatgpt.buffer.data_types import PPORLElement, PromptBatch
from chatgpt.buffer.prompt_pipeline import PromptPipeline
from chatgpt.buffer.replay_buffer import (ExamplesSampler, ExperienceDataset,
                                          Memory)
from chatgpt.buffer.rollout import BaseRolloutStore, PPORolloutStorage
from chatgpt.rlhf.actor_critic import ActorCritic
from chatgpt.rlhf.reward_model import RewardModel
from chatgpt.utils.modeling import flatten_dict, get_tensor_stats, whiten


class RLTrainer:
    def __init__(
        self,
        num_episodes: int = 10,
        max_timesteps: int = 10,
        num_examples: int = 10,
        update_timesteps: int = 10,
        checkpoint_steps: int = 10,
        epochs: int = 10,
        batch_size: int = 32,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        beta_s: float = 0.01,
        actor_eps_clip: float = 0.2,
        critic_eps_clip: float = 0.2,
        device: str = 'cpu',
        debug: bool = False,
    ) -> None:

        self.num_episodes = num_episodes
        self.max_timesteps = max_timesteps
        self.num_examples = num_examples
        self.update_timesteps = update_timesteps
        self.checkpoint_steps = checkpoint_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.beta_s = beta_s
        self.actor_eps_clip = actor_eps_clip
        self.critic_eps_clip = critic_eps_clip
        self.device = device
        self.debug = debug

        # initialize agent-critic
        self.actor_critic = ActorCritic()
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(),
                                          lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(), lr=critic_lr)
        # initialize reward model
        self.reward_model = RewardModel()
        # initialize examples sampler
        self.example_sampler = ExamplesSampler()
        # eps
        self.eps = 1e-8

    def learn(self, memories: Deque[Memory]) -> None:
        """Train the agent-critic model using RL:
        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        print('Start to Learn...')
        # create dataset from memories
        dataloader = DataLoader(ExperienceDataset(memories, self.device),
                                batch_size=self.batch_size)
        # train agent-critic
        self.actor_critic.train()
        for epoch in range(self.epochs):
            for i, batch in enumerate(dataloader):
                (
                    states_actor,
                    old_actions,
                    old_values,
                    rewards,
                    old_actions_log_probs,
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                ) = [tensor.to(self.device) for tensor in batch]

                if self.debug:
                    print(f"#########################################"
                          f" batch from memories {k} \n "
                          f"#########################################"
                          f"states_actor {states_actor.shape} \n"
                          f"old_actions {old_actions.shape} \n"
                          f"old_values {old_values.shape} \n"
                          f"rewards {rewards.shape} \n"
                          f"old_actions_log_probs "
                          f"{old_actions_log_probs.shape}\n"
                          f"sequences_actor {sequences_actor.shape} \n"
                          f"sequences_mask_actor "
                          f"{sequences_mask_actor.shape} \n"
                          f"sequences_critic {sequences_critic.shape} \n"
                          f"sequences_mask_critic "
                          f"{sequences_mask_critic.shape} \n"
                          f"action_len_actor {action_len_actor} \n"
                          f"action_len_critic {action_len_critic} \n"
                          f"#########################################")

                # get actor critic new probabilities and values
                actions_logits, values = self.actor_critic.forward(
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor.item(),
                    action_len_critic.item(),
                )

                # get action log prob
                actions_prob = (torch.softmax(actions_logits,
                                              dim=-1).max(dim=-1).values)
                actions_log_prob = torch.log(actions_prob + self.eps)

                # compute entropy
                entropies = (actions_prob * actions_log_prob).sum(dim=-1)

                # compute KL divergence
                kl_div_loss = (
                    (actions_prob *
                     (old_actions_log_probs - actions_log_prob)).sum(
                         dim=-1).mean())
                # compute PPO Loss -- Whan dimensions are different
                # (especially the values and the probs are
                #  multiplied directly with the reward)
                ratios = (actions_log_prob - old_actions_log_probs).exp()
                advantages = rewards - old_values[:, -1]
                # normalize advantages
                advantages = (advantages - advantages.mean(dim=-1)) / (
                    advantages.std() + self.eps)
                surr1 = advantages * ratios
                surr2 = (torch.clamp(ratios, 1 - self.actor_eps_clip,
                                     1 + self.actor_eps_clip) * advantages)
                policy_loss = -torch.min(surr1,
                                         surr2) - self.beta_s * entropies
                policy_loss = policy_loss.mean()
                loss = policy_loss + kl_div_loss
                # check if loss item is nan
                if torch.isnan(loss):
                    raise ValueError('Loss is nan')
                print('loss', loss.item())

                if self.debug:
                    print('values', values)
                    print('old_values', old_values)
                    print('rewards', rewards)
                    print('ratios', ratios)
                    print('advantages', advantages)
                    print('entropies', entropies)

                # update actor with loss
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                # compute value loss
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -self.critic_eps_clip, self.critic_eps_clip)
                value_loss1 = (value_loss_clipped - rewards)**2
                value_loss2 = (values - rewards)**2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError('Value loss is nan')
                print('value_loss', value_loss.item())

                # upate critic with loss
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        self.actor_critic.eval()
        print('End Learning')
        return policy_loss.item(), value_loss.item(), kl_div_loss.item()

    def train(self) -> None:
        print('Start RL Training')
        # check dimensions consistency
        # at each time step num_examples memories are generated
        number_of_memories_per_learn_iteration = (self.num_examples *
                                                  self.update_timesteps)
        # the number of memories must be a multiple of the batch size
        assert (
            number_of_memories_per_learn_iteration % self.batch_size == 0
        ), 'The number of memories must be a multiple of the batch size'
        # the total number of timesteps is
        total_number_of_timesteps = self.num_episodes * self.max_timesteps
        # the update_timesteps must be a multiple
        #  of the total number of timesteps
        assert total_number_of_timesteps % self.update_timesteps == 0, (
            'The number of timesteps (num_episodes*max_timesteps)'
            'must be a multiple of the update_timesteps')

        # initialize memories
        memories = deque([])
        # initialize counters
        cnt_timesteps = 0
        cnt_learn_iter = 0

        # loop over episodes and timesteps
        self.actor_critic.eval()
        for episode in range(self.num_episodes):
            for timestep in range(self.max_timesteps):
                print(
                    f'Episode: {episode + 1} of {self.num_episodes}, '
                    f'Timestep: {timestep + 1} of {self.max_timesteps}', )
                # counter used to count timesteps into memory
                cnt_timesteps += 1
                # sample num_examples examples from  example dataset
                inputs = self.example_sampler.sample(self.num_examples)
                # tokenize examples
                tokenized_inputs = self.actor_critic.actor.tokenizer(
                    inputs, padding=True, return_tensors='pt')
                if self.debug:
                    print('RLTrainer.train()')
                    print('tokenized inputs', tokenized_inputs)
                # states are [batch_size, seq_len_of_states]
                states = tokenized_inputs['input_ids'].to(self.device)
                states_mask = tokenized_inputs['attention_mask'].to(
                    self.device)

                (actions, actions_logits, values, sequences,
                 sequences_mask) = self.actor_critic.generate(
                     states, states_mask)

                # from action logits to action log probs
                action_prob = (torch.softmax(actions_logits,
                                             dim=-1).max(dim=-1).values)
                actions_log_probs = torch.log(action_prob + self.eps)

                completions = [
                    self.actor_critic.actor.tokenizer.decode(action)
                    for i, action in enumerate(actions)
                ]
                if self.debug:
                    print('RLTrainer.train()')
                    print('completions:')
                    for i, completion in enumerate(completions):
                        print(i, completion)
                        print('')

                task_responses = []
                for input, completion in zip(inputs, completions):
                    task_response = input + '\n' + completion
                    task_responses.append(task_response)
                if self.debug:
                    print('RLTrainer.train()')
                    print('task_responses:')
                    for i, task_response in enumerate(task_responses):
                        print(i, task_response)
                        print('')
                tokenized_responses = self.reward.tokenizer(
                    task_responses, padding=True, return_tensors='pt')
                rewards = self.reward.get_reward(
                    tokenized_responses['input_ids'].to(self.device),
                    tokenized_responses['attention_mask'].to(self.device),
                )

                # store memories of the episode / timestep
                for i in range(states.shape[0]):
                    memories.append(
                        Memory(*map(lambda x: x.detach().cpu(), (
                            states[i, :],
                            actions[i, :],
                            sequences[i, :],
                            values[i, :],
                            rewards[i],
                            actions_log_probs[i, :],
                            sequences_mask[i, :],
                        ))))

                # learn from memories
                print(
                    f'Learning counter: {cnt_timesteps} of {self.update_timesteps}'
                )
                if (cnt_timesteps % self.update_timesteps
                        == 0) and (cnt_timesteps != 0):
                    self.learn(memories)
                    memories.clear()
                    cnt_timesteps = 0
                    cnt_learn_iter += 1

        print('End RL Training')

    def get_advantages_and_returns(
        self,
        values: TensorType['batch_size', 'response_size'],
        rewards: TensorType['batch_size', 'response_size'],
        response_length: int,
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and
        values. Calculated as in the original PPO paper:
        https://arxiv.org/abs/1707.06347 Note that rewards may include a KL
        divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def get_loss(
        self,
        logprobs: TensorType['batch_size', 'response_size'],
        values: TensorType['batch_size', 'response_size'],
        old_logprobs: TensorType['batch_size', 'response_size'],
        old_values: TensorType['batch_size', 'response_size'],
        advantages: TensorType['batch_size', 'response_size'],
        returns: TensorType['batch_size', 'response_size'],
        mask: TensorType['batch_size', 'response_size'],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask)**2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(),
                        clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=n / mask.numel(),
        )

        return loss, flatten_dict(stats)

    def add_prompt_rollout(self, pipeline: BaseRolloutStore):
        """Add a prompt pipeline dataloader to a trainer instance for the
        `make_experience` stage."""
        prompt_dataloader = pipeline.create_loader(self.chunk_size,
                                                   shuffle=True)
        self.prompt_iterator = iter(prompt_dataloader)

    def make_experience(self, num_rollouts: int):
        """Make experience by interacting with the environment.

        Args:
            num_rollouts: Number of rollouts to make

        Returns:
            List of experiences
        """
        experiences = []
        while self.data_rollout.size() < num_rollouts:
            try:
                batch: PromptBatch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = iter(self.prompt_dataloader)
                batch = next(self.prompt_iterator)

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(**batch)
            scores = torch.empty(len(samples))
            prompt_tensors = batch.input_ids
            str_samples, str_prompts, str_outputs = self.decode(
                prompt_tensors, samples)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids

            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                ) for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to()

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(
                scores)

            if self.config.method.scale_reward == 'running':
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == 'ref':
                scores /= self.ref_std

            clip_reward = self.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs),
                                   dim=1)
            attention_mask = all_tokens.not_equal(
                self.tokenizer.pad_token_id).long().to(device)
            with torch.no_grad():
                logits, *_, values = self.model(
                    all_tokens,
                    attention_mask=attention_mask,
                )
                ref_logits = self.ref_model(
                    all_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
                ref_logits = ref_logits.to(device)

            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :],
                                              all_tokens[:, 1:])

        return experiences
