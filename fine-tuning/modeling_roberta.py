from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from  torch.nn.utils.rnn  import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

class RobertaForTokenClassification_v2(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size*2, config.num_labels)
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.softmax = nn.Softmax(dim=2)

        self.lambda1 = 1e-1
        self.lambda2 = 1e-3
        self.epsilon = 1e-8
        self.threshold = 0.3

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        entity_ids=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        device = input_ids.device

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)

        logits = self.classifier(sequence_output)
        """ Bilstm for label refinement """
        if entity_ids is not None:
            entity_ids = entity_ids[:,:,None]
            bilstm_hidden = self.rand_init_hidden(batch_size)
            fst_bilstm_hidden = bilstm_hidden[0].to(device)
            bst_bilstm_hidden = bilstm_hidden[1].to(device)

            lstm_out, lstm_hidden = self.bilstm(sequence_output, (fst_bilstm_hidden, bst_bilstm_hidden))
            lstm_out = lstm_out.contiguous().view(-1, self.config.hidden_size*2)
            d_lstm_out = self.dropout(lstm_out)
            l_out = self.classifier2(d_lstm_out)
            lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)

            """ make label representation similar on biomedical entities (without regarding to context representation) """
            sft_logits = self.softmax(logits)
            sft_feats = self.softmax(lstm_feats)
            kl_logit_lstm = F.kl_div(sft_logits.log(), sft_feats, None, None, 'sum')
            kl_lstm_logit = F.kl_div(sft_feats.log(), sft_logits, None, None, 'sum')
            kl_distill = (kl_logit_lstm + kl_lstm_logit) / 2

            """ update entities with lstm and mlp classifier """
            sft_feats = sft_feats * entity_ids # mask for only updated entities
            
            """ update through uncertainties """
            uncertainty = -torch.sum(sft_logits * torch.log(sft_logits + self.epsilon), dim=2)
            ones = torch.ones(uncertainty.shape).to(device)
            zeros = torch.zeros(uncertainty.shape).to(device)
            uncertainty_mask = torch.where(uncertainty > self.threshold, ones, zeros)
            uncertainty_mask = uncertainty_mask[:,:,None]
            sft_feats = sft_feats * uncertainty_mask

            logits = logits + sft_feats

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]

            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if entity_ids is not None:
                active_lstm_logits = sft_feats.view(-1, self.num_labels)[active_loss]
                lstm_loss = loss_fct(active_lstm_logits, active_labels)
                final_loss = loss + (self.lambda1) * lstm_loss + (self.lambda2) * kl_distill
                outputs = (final_loss,) + outputs
            else:
                outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)

    def rand_init_hidden(self, batch_size,):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2 * 2, batch_size, self.config.hidden_size)), Variable(torch.randn(2 * 2, batch_size, self.config.hidden_size))

