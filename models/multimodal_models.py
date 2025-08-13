from typing import List, Optional, Tuple, Union

import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, BertConfig, BertModel

from models import vision_models
from models.loss_fcts import get_loss_function


def setup_llm(llm_config, lora_config=None):
    llm_module = getattr(transformers, llm_config["model_class"])
    model = llm_module.from_pretrained(
        llm_config["model_id"],
    )
    # if llm_config.get("embedding_only", False):
    #    return EmbedOnlyModel(model)

    if llm_config.get("bert_encoder_layers", None):
        n_layers = llm_config["bert_encoder_layers"]
        model.encoder.layer = model.encoder.layer[:n_layers]
        return model
        # return BertEncoderModel(model, n_layers)

    if lora_config is not None:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
    return model


def setup_vision(vision_config):
    model_module = getattr(vision_models, vision_config["model_class"])
    model = model_module(vision_config)
    return model


class ConcatLayer(nn.Module):
    def forward(self, llm_embeds, vis_embeds, attention_mask, position_ids):
        mm_embeds = torch.cat([vis_embeds, llm_embeds], dim=1)
        batchsize, vseqlen = vis_embeds.size(0), vis_embeds.size(1)
        if attention_mask is not None:
            # check if to set attention_mask as ones if given None
            padding_ = torch.ones(
                (batchsize, vseqlen),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((padding_, attention_mask), dim=1)

        if position_ids is not None:
            vis_pos_ids = torch.arange(
                0, vseqlen, dtype=position_ids.dtype, device=position_ids.device
            )
            vis_pos_ids = vis_pos_ids.unsqueeze(0).expand(batchsize, -1)
            position_ids = torch.cat((vis_pos_ids, position_ids), dim=1)
        return mm_embeds, attention_mask, position_ids


def _pos_id_from_embeds(embeds):
    pos_ids = torch.arange(0, embeds.size(1), device=embeds.device, dtype=torch.long)
    return pos_ids.unsqueeze(0).expand(embeds.size(0), -1)


def setup_mixer(mixer_config):
    if mixer_config is None or mixer_config["name"] == "concatenate":
        return ConcatLayer()


class Uniferum(BertModel):
    config_class = BertConfig

    def __init__(self, config):
        # this is annoying, fixme
        llm_config = AutoConfig.from_pretrained(config["llm_args"]["model_id"])
        super().__init__(llm_config)
        self.config = llm_config
        self.setup_config = config
        self.model = setup_llm(config["llm_args"], lora_config=config.get("lora_args"))
        self.vision_model = setup_vision(config["vision_args"])
        self.mixer = setup_mixer(config.get("mixer_args"))
        self.vocab_size = llm_config.vocab_size
        self.lm_head_cls = nn.Linear(llm_config.hidden_size, 1)
        self.lm_head_seg = nn.Linear(llm_config.hidden_size, 64)
        self.loss_fct_cls = get_loss_function(
            config.get("loss_fct_cls", "BCEWithLogitsLoss"), reduction="none"
        )
        self.loss_fct_seg = get_loss_function(
            config.get("loss_fct_seg", "SigmoidFocalLoss"), reduction="none"
        )
        self.post_init()

    def get_multimodal_embeds(self, input_ids, position_ids, attention_mask, images):
        vis_embeds = self.vision_model(images)  # [batch, seqlen0, fea_dim]
        lm_embeds = self.model.embeddings(input_ids)  # [batch, seqlen1, fea_lm]
        v_batch = vis_embeds.size(0)
        lm_batch = lm_embeds.size(0)
        if v_batch == 1 and lm_batch > 1:
            vis_embeds = vis_embeds.expand(lm_batch, -1, -1, -1)

        # need to handle attention
        mm_embeds, attention_mask, position_ids = self.mixer(
            lm_embeds, vis_embeds, attention_mask, position_ids
        )

        # TODO improve position ids
        position_ids = _pos_id_from_embeds(mm_embeds)
        return None, position_ids, attention_mask, mm_embeds

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seg_label=None,
        task_mask=None,
        **kwargs,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds,
        ) = self.get_multimodal_embeds(
            input_ids,
            position_ids,
            attention_mask,
            images,
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        logits_cls = self.lm_head_cls(last_hidden_state[:, 0])
        logits_seg = self.lm_head_seg(last_hidden_state[:, 1:161])
        if labels is not None:
            cls_losses = self.loss_fct_cls(logits_cls, labels)
            seg_losses = self.loss_fct_seg(logits_seg, seg_label)
            losses = task_mask * cls_losses.mean(dim=1) + 10.0 * (
                1 - task_mask
            ) * seg_losses.mean(dim=2).mean(dim=1)
            loss = losses.mean()
        else:
            return logits_cls

        if not return_dict:  ## not tested
            output = (logits_cls,) + transformer_outputs
            return ((loss,) + output) if loss is not None else output
        return {"loss": loss, "logits": logits_cls, "logits_seg": logits_seg}
