from typing import List, Optional, Tuple, Union
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from transformers.adapters.modeling import Adapter
from transformers.adapters import (
    BartAdapterModel,
    RobertaAdapterModel,
    BertAdapterModel,
    AdapterConfig,
)
import torch
import code
from torch import nn
from torch.nn import MSELoss

from torch.autograd import Function

patch_typeguard()


class StopGradient(Function):
    @staticmethod
    def forward(ctx, i):
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return -0 * grad_output


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, i):
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return -0.1 * grad_output


@typechecked
class AlignmentMixin(nn.Module):
    def __init__(self, config):
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        super().__init__(config)

        self.critic_transform = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size, nhead=12
        )
        self.critic_score = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, 1),
        )

        self.counter = 0

    @torch.no_grad()
    def produce_original_embeddings(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
    ) -> TensorType["batch", "seq_len", "hidden_size"]:
        self.train(False)
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        if "hidden_states" in outputs:
            hidden_mat = torch.stack(
                [
                    hidden_state[:, :, :] * attention_mask.unsqueeze(-1)
                    for hidden_state in outputs.hidden_states[-1:]
                ],
                dim=1,
            )
        else:
            hidden_mat = torch.stack(
                [
                    hidden_state[:, :, :] * attention_mask.unsqueeze(-1)
                    for hidden_state in outputs.encoder_hidden_states[-1:]
                ],
                dim=1,
            )
        self.train(True)
        return hidden_mat.squeeze(1)

    def critic(self, embedding):
        mask = embedding.sum(-1) != 0
        cls_token = self.critic_transform(
            embedding.permute(1, 0, 2), src_key_padding_mask=mask
        )[0, :, :].squeeze()
        print(cls_token.shape)
        scores = self.critic_score(cls_token)
        return scores.mean()

    def forward(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        original_embedding: Optional[
            TensorType["batch", "layers", "hidden_size"]
        ] = None,
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
        **kwargs
    ):
        if type(original_embedding) != type(None):
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            if "hidden_states" in outputs:
                hidden_mat = torch.stack(
                    [
                        hidden_state[:, :, :] * attention_mask.unsqueeze(-1)
                        for hidden_state in outputs.hidden_states[-1:]
                    ],
                    dim=1,
                )
            else:
                hidden_mat = torch.stack(
                    [
                        hidden_state[:, :, :] * attention_mask.unsqueeze(-1)
                        for hidden_state in outputs.encoder_hidden_states[-1:]
                    ],
                    dim=1,
                )
            hidden_mat = GradientReversal.apply(hidden_mat.squeeze(1))
            alignment_loss = (original_embedding[:, 0, :] - hidden_mat[:, 0, :]).square().sum(1).mean()
            loss = self.critic(hidden_mat) - self.critic(original_embedding)
            print(
                alignment_loss,
                loss,
            )
            self.counter += 1
            return (loss-alignment_loss,)


@typechecked
class BartAdapterModelForAlignment(AlignmentMixin, BartAdapterModel):
    def __init__(self, config):
        config.dropout = 0.0
        config.activation_dropout = 0.0
        config.attention_dropout = 0.0
        config.classifier_dropout = 0.0
        super().__init__(config)


@typechecked
class RobertaAdapterModelForAlignment(AlignmentMixin, RobertaAdapterModel):
    def __init__(self, config):
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        super().__init__(config)


@typechecked
class BertAdapterModelForAlignment(AlignmentMixin, BertAdapterModel):
    def __init__(self, config):
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        super().__init__(config)
