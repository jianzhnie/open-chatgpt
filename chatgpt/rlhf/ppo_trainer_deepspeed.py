import torch
import torch.nn.functional as F
from torch.optim import Adam

from chatgpt.rlhf.actor_critic import ActorModel, CriticModel
from chatgpt.rlhf.reward_model import RewardModel


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class PPOTrainer():

    def __init__(
        self,
        pretrained: str = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        kl_ctl: float = 0.02,
        clip_reward_value: float = 5,
        cliprange: float = 0.2,
        cliprange_value: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_answer_seq_len: int = 256,
        device: str = 'cpu',
    ):
        self.actor_model = ActorModel(pretrained=pretrained).to(device)
        self.critic_model = CriticModel(pretrained=pretrained).to(device)
        self.ref_model = ActorModel(pretrained=pretrained).to(device)
        self.reward_model = RewardModel(pretrained=pretrained).to(device)
        self.tokenizer = self.actor_model.tokenizer

        # Those value can be changed
        self.lam = lam
        self.gamma = gamma
        self.kl_ctl = kl_ctl
        self.clip_reward_value = clip_reward_value
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.max_answer_seq_len = max_answer_seq_len

        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic_model.parameters(),
                                     lr=critic_lr)
        self.device = device

    def _generate_sequence(self, prompts):
        prompts = prompts.to(self.device)
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        with torch.no_grad():
            seq = self.actor_model.model.generate(prompts,
                                                  max_length=max_min_length,
                                                  min_length=max_min_length)

        # Filter out seq with no asnwers (or very short). This happens when users directly \
        # use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    def generate_experience(self, prompts):
        self.eval()
        seq = self._generate_sequence(prompts)
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            logits = self.actor_model(seq, attention_mask=attention_mask)
            logits_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        exp_dict = {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            'attention_mask': attention_mask
        }
        for key, value in exp_dict.items():
            print(key, value.shape, value.dtype)
        return exp_dict

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        (prompts, log_probs, ref_log_probs, reward_score, seq, attention_mask,
         values) = [tensor.to(self.device) for tensor in inputs]
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        # process the new outputs
        batch = {'input_ids': seq, 'attention_mask': attention_mask}
        actor_prob = self.actor_model.forward(**batch)
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        # policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        # value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.training
        assert self.critic_model.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.training
        assert not self.critic_model.training
        assert not self.ref_model.training
        assert not self.reward_model.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()
