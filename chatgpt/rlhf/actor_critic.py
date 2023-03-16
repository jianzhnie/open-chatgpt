from typing import Optional, Tuple
from collections import namedtuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from einops.layers.torch import Rearrange

ActionCriticReturn = namedtuple(
    'ActionCriticReturn',
    ['actions', 'action_logits', 'values', 'sequence', 'sequences_mask'])


class ActorModel(nn.Module):
    """
    Actor model that generates logits representing the probability distribution over the vocabulary of actions.

    Args:
        pretrained (str, optional): Pretrained model name or path.
        debug (bool, optional): Whether to print debug information. Defaults to False.
    """
    def __init__(self, pretrained: Optional[str] = None, debug: bool = False):
        super().__init__()

        # Load tokenizer and set special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained,
                                                       padding_side='left')
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.eos_token_id = 0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load pre-trained language model
        self.model = AutoModelForCausalLM.from_pretrained(pretrained)

        self.debug = debug

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate logits to have probability distribution over the vocabulary of the actions.

        Args:
            input_ids (torch.Tensor): Sequences of states and actions used to compute token logits for the whole list of sequences.
            attention_mask (torch.Tensor): Mask for the sequences attention.

        Returns:
            logits (torch.Tensor): Logits for the actions taken.
        """
        model_output = self.model(input_ids, attention_mask=attention_mask)
        # Extract logits from the model output
        logits = model_output.logits

        if self.debug:
            print('ActorModel.forward')
            print('logits shape:', logits.shape)
            print('logits:', logits)

        return logits

    @torch.no_grad()
    def generate(self, states: torch.Tensor, state_mask: torch.Tensor,
                 temperature, max_sequence_length,
                 max_tokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate actions and sequences=[states, actions] from state (i.e.
        input of the prompt generator model)

        Args:
            states (torch.Tensor): Input sequence tensor with only state IDs.
            state_mask (torch.Tensor): Attention mask for input state tensor.
            temperature (float): Softmax temperature to apply during generation.
            max_sequence_length (int): Maximum allowed length of generated sequence.
            max_tokens (int): Maximum number of tokens to generate after input sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of generated actions and full generated sequences.
        """
        # Set maximum length of generation
        max_generation_possible = max_sequence_length - states.shape[1]
        max_completion = min(max_tokens, max_generation_possible)
        if max_completion <= 0:
            raise ValueError(
                'The maximum completion available is <= 0 the prompt is too long w.r.t the model sequence length'
            )
        max_length = states.shape[1] + max_completion

        # Generate actions and sequences
        sequences = self.model.generate(
            input_ids=states,
            attention_mask=state_mask,
            temperature=temperature,
            max_length=max_length,
        )
        actions = sequences[:, states.shape[
            1]:]  # Extract generated actions from full sequence
        if self.debug:
            print('ActorModel.generate')
            print('state', states)
            print('state shape', states.shape)
            print('sequence shape', sequences.shape)
            print('sequence', sequences)
            print('actions shape', actions.shape)
            print('actions', actions)
        return actions, sequences


class CriticModel(nn.Module):
    """Critic model that evaluates the quality of a given sequence of tokens.

    Args:
        pretrained (str): Pretrained model name or path.
        debug (bool): Whether to print debugging information or not.
    """

    def __init__(self,
                 model="opt",
                 pretrained: Optional[str] = None,
                 debug: bool = True):
        super().__init__()

        # Instantiate tokenizer and model from pretrained checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained,
                                                       padding_side="left",
                                                       truncation_side="left")
        self.model = AutoModel.from_pretrained(pretrained)

        # Set EOS token and padding token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
            self.tokenizer.eos_token_id = 0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.debug = debug
        self.config = self.model.config

        # Define value head layers to output a scalar value
        if model == "opt":
            head_hidden_size = self.config.word_embed_proj_dim
        else:
            head_hidden_size = self.config.head_hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 1),
            Rearrange("... 1 -> ..."),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Evaluate the quality of a sequence of tokens.

        Args:
            input_ids (torch.Tensor): Tensor of token ids of shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Tensor of rewards of shape (batch_size, 1)
        """
        output = self.model(input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
        rewards = self.value_head(output.last_hidden_state)

        # Print debugging information
        if self.debug:
            print("CriticModel.forward")
            print("input_ids.shape", input_ids.shape)
            print("input_ids", input_ids)
            print("rewards.shape", rewards.shape)
            print("rewards", rewards)

        return rewards

    def get_reward(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the reward for a sequence of tokens.

        Args:
            input_ids (torch.Tensor): Tensor of token ids of shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Tensor of rewards of shape (batch_size,)
        """
        rewards = self.forward(input_ids, attention_mask)
        return rewards[:, -1]


class ActorCritic(nn.Module):
    """Actor Critic class stores both the actor and the critic models and it
    generates values and action for given sequences during the training of the
    actor.

    Args:
        actor (nn.Module): Actor model
        critic (nn.Module): Critic model
        debug (bool): enable prints for Debugging

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        debug: bool = False,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.debug = debug

    def forward(
        self,
        sequences: torch.Tensor,
        sequences_mask: torch.Tensor,
        action_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given the whole sequences, use the actor forward to get the logits
        for each token in the sequence and the critic forward to get the values
        for each generation step.

        Args:
            sequences (torch.Tensor): Sequences composed of [states, actions]
            sequences_mask (torch.Tensor): Mask for the sequences
            action_len (int): Length of the actions in the sequences

        Returns:
            action_logits (torch.Tensor): Logits for the actions in the
                sequences
            values (torch.Tensor): Values for the actions in the sequences
        """
        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        actions_logits = self.actor(sequences, sequences_mask)
        values = self.critic(sequences, sequences_mask)

        # return only logits and values for the actions taken
        real_actions_logits = actions_logits[:, -action_len:, :]
        real_values = values[:, -action_len:]

        if self.debug:
            print('ActorCritic.forward')
            print('action_len', action_len)
            print('sequences.shape', sequences.shape)
            print('sequences', sequences)
            print('real_action_logits.shape', real_actions_logits.shape)
            print('real_action_logits', real_actions_logits)
            print('real_values.shape', real_values.shape)
            print('real_values', real_values)

        return real_actions_logits, real_values

    @torch.no_grad()
    def generate(self, states: torch.Tensor,
                 state_mask: torch.Tensor) -> ActionCriticReturn:
        """Generate actions, action_logits, values, and sequences from states.

        Args:
            states (torch.Tensor): The states of the environment.
            state_mask (torch.Tensor): The mask for the states of the environment.

        Returns:
            ActionCriticReturn: A namedtuple containing the following fields:
                - actions (torch.Tensor): The generated actions.
                - actions_logits (torch.Tensor): The logits for the generated actions.
                - values (torch.Tensor): The values generated by the critic model
                for the actions generated by the actor.
                - sequences (torch.Tensor): The sequences generated from the states as
                [states, actions].
                - sequences_mask (torch.Tensor): The mask for the generated sequences.

        """
        # Generate action sequence.
        actions, sequence = self.actor.generate(states, state_mask)

        # Get the mask for the generated sequence.
        sequences_mask = sequence != self.actor.tokenizer.pad_token_id
        sequences_mask = sequences_mask.to(sequence.device).long().detach()

        # Get the length of the generated actions.
        action_len = actions.shape[1]

        # Generate action logits and values.
        actions_logits, values = self.forward(sequence, sequences_mask,
                                              action_len)

        if self.debug:
            print('ActorCritic.generate')
            print('actions shape', actions.shape)
            print('actions', actions)
            print('sequence shape', sequence.shape)
            print('sequence', sequence)
            print('actions_logits shape', actions_logits.shape)
            print('actions_logits', actions_logits)
            print('values shape', values.shape)
            print('values', values)

        return ActionCriticReturn(actions, actions_logits, values, sequence,
                                  sequences_mask)
