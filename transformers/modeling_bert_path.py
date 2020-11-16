import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertLMPredictionHead,
)


class BertForPathLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.seq_score = nn.Linear(config.hidden_size, 2)
        # self.nspcls = BertOnlyNSPHead(config)
        # self.scls = BertOnlyNSPHead(config)

        self.weight_loss_mlm = 1.0
        self.weight_loss_nsp = 20.0
        self.weight_loss_sp = 20.0

        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        element_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        next_sentence_label=None,
        sequence_label=None,
        same_node_labels=None,
        lm=True,
        nsp=True,
        sp=True,
        sametoken=True
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            element_type_ids=element_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        # outputs = outputs[2:]
        scores = dict()
        if lm:
            prediction_scores = self.predictions(sequence_output)
            scores['mlm'] = prediction_scores
        if nsp:
            seq_relationship_score = self.seq_relationship(pooled_output)
            scores['nsp'] = seq_relationship_score
        if sp:
            seq_score = self.seq_score(pooled_output)
            scores['sp'] = seq_score
        if sametoken:
            pass

        has_gt = False
        # total_loss = torch.zeros().to(device)

        losses = dict()
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            losses['mlm'] = masked_lm_loss * self.weight_loss_mlm
            has_gt = True
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            losses['nsp'] = next_sentence_loss * self.weight_loss_nsp
            has_gt = True
        if sequence_label is not None:
            loss_fct = CrossEntropyLoss()
            sequence_loss = loss_fct(seq_score.view(-1, 2), sequence_label.view(-1))
            losses['sp'] = sequence_loss * self.weight_loss_sp
            has_gt = True
        if same_node_labels is not None:
            has_gt = True
            pass

        if has_gt:
            total_loss = torch.stack(list(losses.values()), dim=-1).sum(dim=-1)
        else:
            total_loss = None

        return total_loss, losses, scores, outputs[2:]  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
