# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from pytorch_transformers.modeling_xlnet import SequenceSummary


class XLNET_SPC(nn.Module):
    def __init__(self, transformer, opt):
        super(XLNET_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.transformer = transformer
        self.sequence_summary = SequenceSummary(transformer.config)
        self.logits_proj = nn.Linear(transformer.config.d_model, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        output = self.transformer(text_bert_indices, bert_segments_ids)[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        return logits
