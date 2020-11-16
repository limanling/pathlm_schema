import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .modeling_utils import SequenceSummary


from .modeling_xlnet import (
    XLNetModel,
    XLNetPreTrainedModel,
    XLNetLMHeadModel,
    XLNetForSequenceClassification,
)

class XLNetForPathLM(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # from XLNetLMHeadModel
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)

        # from XLNetForSequenceClassification
        self.num_labels = config.num_labels
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.weight_loss_clm = 50.0
        self.weight_loss_nsp = 1.0

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_loss

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # Add dummy token at the end (no attention on this one)

        effective_batch_size = input_ids.shape[0]
        dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # Build permutation mask so that previous tokens don't see last token
        sequence_length = input_ids.shape[1]
        perm_mask = torch.zeros(
            (effective_batch_size, sequence_length, sequence_length), dtype=torch.float, device=input_ids.device
        )
        perm_mask[:, :, -1] = 1.0

        # We'll only predict the last token
        target_mapping = torch.zeros(
            (effective_batch_size, 1, sequence_length), dtype=torch.float, device=input_ids.device
        )
        target_mapping[0, 0, -1] = 1.0

        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_cache": kwargs["use_cache"],
        }

        # if past is defined in model kwargs then use it for faster decoding
        if past:
            inputs["mems"] = past

        return inputs

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        element_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels_clm=None,
        labels_seq=None,
        lm=False,
        nsp=False,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            element_type_ids=element_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        output = transformer_outputs[0]

        scores = dict()
        # sp
        if nsp:
            output_seq = self.sequence_summary(output)
            logits_seq = self.logits_proj(output_seq)
            scores['nsp'] = logits_seq
        # clm
        if lm:
            logits_clm = self.lm_loss(output)
            # log_logits_clm = F.softmax(logits_clm, dim=-1)  # shape (bsz, slen, start_n_top)
            # print('logits_clm', log_logits_clm)
            scores['clm'] = logits_clm

        # outputs = (logits_seq,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        # outputs = (logits_clm,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        losses = dict()
        has_gt = False
        if labels_seq is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                sequence_loss = loss_fct(logits_seq.view(-1), labels_seq.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                sequence_loss = loss_fct(logits_seq.view(-1, self.num_labels), labels_seq.view(-1))
            losses['nsp'] = sequence_loss * self.weight_loss_nsp
            has_gt = True
            # outputs = (loss,) + outputs
        if labels_clm is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_clm = loss_fct(logits_clm.view(-1, logits_clm.size(-1)), labels_clm.view(-1))
            losses['clm'] = loss_clm * self.weight_loss_clm
            has_gt = True
            # outputs = (loss,) + outputs


        if has_gt:
            total_loss = torch.stack(list(losses.values()), dim=-1).sum(dim=-1)
        else:
            total_loss = None

        # return outputs  # return (loss), logits, (mems), (hidden states), (attentions)
        # return outputs  # return (loss), logits, (mems), (hidden states), (attentions)
        return total_loss, losses, scores, transformer_outputs[1:]


