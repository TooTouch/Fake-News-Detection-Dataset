from transformers import AutoConfig, BertModel, BertPreTrainedModel
import torch.nn as nn

from .registry import register_model
from .utils import download_weights

class BTS(BertPreTrainedModel):
    def __init__(self, pretrained_name, config, num_classes):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(pretrained_name, config=config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask       = attention_mask,
            token_type_ids       = token_type_ids,
            position_ids         = position_ids,
            head_mask            = head_mask,
            inputs_embeds        = inputs_embeds,
            output_attentions    = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_attentions:
            return logits, outputs[-1]
        else:
            return logits

@register_model
def bts(hparams, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = BTS(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes']
    )

    return model

@register_model
def bts_task1(pretrained=False, **kwargs):
    # pretrained weights
    url = 'https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/weights/BTS_task1.pt'
    
    model_config = AutoConfig.from_pretrained('klue/bert-base')
    model = BTS(
            pretrained_name = 'klue/bert-base', 
            config          = model_config,
            num_classes     = 2
    )

    if pretrained:
        weights = download_weights(url)
        model.load_state_dict(weights)
    
    return model

