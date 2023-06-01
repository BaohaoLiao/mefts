from typing import List, Optional, Tuple, Union
import sys

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.autograd import Function

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertAttention,
    BertOutput,
    BertLayer,
    load_tf_weights_in_bert,
    BertEmbeddings,
    BertPooler,
    BERT_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    _SEQ_CLASS_EXPECTED_OUTPUT,
    _SEQ_CLASS_EXPECTED_LOSS,
)
from .modules import Adapter


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"


class RevBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        if config.f_arch == "attn":
            self.attn_adapter = Adapter(config)
        else:
            self.attn_adapter = None

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.attn_adapter is not None: # For MEFT3
            hidden_states = hidden_states + self.attn_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RevBertAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.output = RevBertSelfOutput(config)


class RevBertOutput(BertOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, adapter_change=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # For MEFT1 and MEFT2: inserting an adapter in parallel to two consecutive feed-forward layers
        if adapter_change is not None:
            hidden_states = hidden_states + adapter_change
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RevBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.output = RevBertOutput(config)
        self.ffn_adapter = Adapter(config)
        if config.f_arch == "attn":  # For MEFT3: insert an adapter to the attention block
            self.rev_adapter = None
            self.attention = RevBertAttention(config)
        else:  # For MEFT1 and MEFT2
            self.rev_adapter = Adapter(config, layernorm=config.layernorm_in_adapter)

        self.x1_factor = config.x1_factor  # by default, lambda = 0.1
        self.x2_factor = config.x2_factor  # by default, beta = 0.1
        self.f_arch = config.f_arch  # What is the choice for the F architecture
        if config.f_arch == "layer":  # MEFT1
            self.F = self.forward_layer
            self.G = self.forward_adapter
        elif config.f_arch == "adapter":  # MEFT2
            self.F = self.forward_adapter
            self.G = self.forward_layer
        elif config.f_arch == "attn":  # MEFT3
            self.F = self.forward_attention
            self.G = self.forward_mlp
        self.seeds = {}

    def set_seed(self, key):
        """
        Record the seed during the forward pass, so the activation can be recomputed.
        """
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(seed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        x1, x2 = torch.chunk(hidden_states, 2, dim=-1)
        if self.training:
            self.set_seed("F")
        f_outputs = self.F(
            x2,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions
        )
        if isinstance(f_outputs, tuple):
            f_x2 = f_outputs[0]
            outputs = f_outputs[1:]
        else:
            f_x2 = f_outputs

        y1 = self.x1_factor * x1 + f_x2

        if self.training:
            self.set_seed("G")
        g_outputs = self.G(
            y1,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions
        )
        if isinstance(g_outputs, tuple):
            g_y1 = g_outputs[0]
            outputs = g_outputs[1:]
        else:
            g_y1 = g_outputs

        y2 = self.x2_factor * x2 + g_y1

        if self.f_arch == "attn":
            y = torch.cat([y1, y2], dim=-1)  # Don't switch for MEFT3
        else:
            y = torch.cat([y2, y1], dim=-1)  # Switch for MEFT1 and MEFT2
        outputs = (y,) + outputs

        return outputs

    def backward_pass(
        self,
        y,
        dy,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value
    ):
        assert self.training, (
            "If you want to train `ReversibleModel` and its variations, make sure to use `model.train()` to put the"
            " model into training mode."
        )
        if self.f_arch == "attn":
            y1, y2 = torch.chunk(y, 2, dim=-1)  # Don't Switch for MEFT3
            dy1, dy2 = torch.chunk(dy, 2, dim=-1)
        else:
            y2, y1 = torch.chunk(y, 2, dim=-1)  # Swich for MEFT1 and MEFT2
            dy2, dy1 = torch.chunk(dy, 2, dim=-1)
        with torch.enable_grad():
            y1.requires_grad = True
            torch.manual_seed(self.seeds["G"])
            g_outputs = self.G(
                y1,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=False
            )
            if isinstance(g_outputs, tuple):
                g_y1 = g_outputs[0]
            else:
                g_y1 = g_outputs

            g_y1.backward(dy2, retain_graph=True)

        with torch.no_grad():
            x2 = (y2 - g_y1) / self.x2_factor
            # save memory
            del g_y1, y2
            dy1 += y1.grad
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            torch.manual_seed(self.seeds["F"])
            f_outputs = self.F(
                x2,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=False
            )
            if isinstance(f_outputs, tuple):
                f_x2 = f_outputs[0]
            else:
                f_x2 = f_outputs
            f_x2.backward(dy1, retain_graph=False)

        with torch.no_grad():
            x1 = (y1 - f_x2) / self.x1_factor
            del f_x2, y1
            dy2 *= self.x2_factor
            dy2 += x2.grad
            x2.grad = None
            dy1 *= self.x1_factor
            x2 = x2.detach()

        return torch.cat([x1, x2], dim=-1), torch.cat([dy1, dy2], dim=-1)

    def forward_adapter(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        return self.rev_adapter(hidden_states)

    def forward_layer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        return self_attention_outputs

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        hidden_states = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, hidden_states
        )
        return hidden_states

    def feed_forward_chunk(self, attention_output):
        if self.ffn_adapter is not None:
            adapter_change = self.ffn_adapter(attention_output)
        else:
            adapter_change = None

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, adapter_change=adapter_change)
        return layer_output


class RevBackProp(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        layers,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        output_attentions,
    ):
        for i, layer_module in enumerate(layers):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

        ctx.save_for_backward(hidden_states.detach())
        ctx.layers = layers
        ctx.attention_mask = attention_mask
        return hidden_states

    def backward(ctx, dy):
        y, = ctx.saved_tensors
        layers = ctx.layers
        attention_mask = ctx.attention_mask
        for i, layer_module in enumerate(layers[::-1]):
            y, dy = layer_module.backward_pass(
                y,
                dy,
                attention_mask=attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None
            )
        return dy, None, None, None, None, None, None, None


class RevBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RevBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.num_rev_layers = config.num_rev_layers
        assert self.num_rev_layers <= config.num_hidden_layers

    def vanilla_backward(
        hidden_states,
        layers,
        attention_mask
    ):
        for i, layer_module in enumerate(layers):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # Set h_0^1 = h_1^1 = h_0
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)

        # Layers with reversible gradient
        layers_for_reverse_backward = self.layer[:self.num_rev_layers]
        # Layers with vanilla gradient
        layers_for_vanilla_backward = self.layer[self.num_rev_layers:]
        if len(layers_for_reverse_backward) == 0:  # In this case, MEFTs are simply PEFT methods
            executing_fn = RevBertEncoder.vanilla_backward
            hidden_states = executing_fn(
                hidden_states,
                layers_for_vanilla_backward,
                attention_mask
            )
        elif len(layers_for_reverse_backward) == self.config.num_hidden_layers:  # All layers are reversible
            executing_fn = RevBackProp.apply
            hidden_states = executing_fn(
                hidden_states,
                layers_for_reverse_backward,
                attention_mask,
                None,
                None,
                None,
                None,
                False
            )
        else:  # The first setting in Figure 7
            lower_executing_fn = RevBackProp.apply
            hidden_states = lower_executing_fn(
                hidden_states,
                layers_for_reverse_backward,
                attention_mask,
                None,
                None,
                None,
                None,
                False
            )
            higher_executing_fn = RevBertEncoder.vanilla_backward
            hidden_states = higher_executing_fn(
                hidden_states,
                layers_for_vanilla_backward,
                attention_mask
            )

        x1, x2 = torch.chunk(hidden_states, 2, dim=-1)
        hidden_states = (x1 + x2) / 2.

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )


class RevBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RevBertEncoder):
            module.gradient_checkpointing = value



class RevBertModel(RevBertPreTrainedModel):
    """
    Only set self.encoder = RevBertEncoder(config) compared to BertModel
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = RevBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



class RevBertForSequenceClassification(RevBertPreTrainedModel):
    """
    Only set self.bert = RevBertModel(config) compared to BertForSequenceClassification
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = RevBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


