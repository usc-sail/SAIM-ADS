import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverClassificationDecoder,
)

class Perceiver_TextVisual_hf_model(nn.Module):
    def __init__(self,d_model,d_latents,num_labels,
                num_self_attends_per_block,num_self_attention_heads,num_cross_attention_heads):
        
        super().__init__()

        self.d_model=d_model
        self.d_latents=d_latents
        self.num_labels=num_labels
        self.num_self_attends_per_block=num_self_attends_per_block
        self.num_self_attention_heads=num_self_attention_heads
        self.num_cross_attention_heads=num_cross_attention_heads

        self.perceiver_config = PerceiverConfig(d_model=self.d_model,
                                                num_labels=self.num_labels,
                                                num_self_attends_per_block=num_self_attends_per_block,
                                                num_self_attention_heads=num_self_attention_heads,
                                                )

        self.decoder=PerceiverClassificationDecoder(self.perceiver_config,
                num_channels=self.perceiver_config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=self.perceiver_config.d_latents, index_dims=1),
                use_query_residual=True)

        self.model=PerceiverModel(self.perceiver_config,decoder=self.decoder)

    def forward(self,inputs,mask,queries=None):
        #basic parameters

        #get the logits
        outputs=self.model(inputs,attention_mask=mask)

        logits=outputs.logits
        #return the logits
        return logits

d_model=256
d_latents=256
num_self_attends_per_block=4
num_self_attention_heads=8
num_cross_attention_heads=8
num_labels=10
num_args={'d_model':d_model,
          'd_latents':d_latents,
          'num_labels':num_labels,
          'num_self_attends_per_block':num_self_attends_per_block,
          'num_self_attention_heads':num_self_attention_heads,
          'num_cross_attention_heads':num_cross_attention_heads
          }

perceiver_text_visual_hf=Perceiver_TextVisual_hf_model(**num_args)
#print(perceiver_text_visual_hf)
inputs=torch.randn((32,256,256))
mask=torch.ones((32,256))
logits=perceiver_text_visual_hf(inputs=inputs,mask=mask)
print(logits.shape)



