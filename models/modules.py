import torch
import torch.nn as nn

from transformers.activations import ACT2FN

# Adapter architecture
class Adapter(nn.Module):
    '''
    The adapter architecture is borrowed from https://arxiv.org/abs/1902.00751
    '''
    def __init__(self, config, layernorm=False):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.adapter_bottleneck_dim)
        self.dense2 = nn.Linear(config.adapter_bottleneck_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        if layernorm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states
