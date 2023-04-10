import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverClassificationDecoder,
)


d_model=256
d_latents=256
num_labels=10
num_self_attends_per_block=4
num_self_attention_heads=8
num_cross_attention_heads=8
num_classes=10

perceiver_config = PerceiverConfig(d_model=d_model,
                    d_latents=d_latents,
                    num_labels=num_classes, 
                    num_self_attends_per_block=num_self_attends_per_block,
                    num_self_attention_heads=num_self_attention_heads,
                    num_cross_attention_heads=4)
print(perceiver_config)
decoder=PerceiverClassificationDecoder(perceiver_config,
                num_channels=perceiver_config.d_latents,
                trainable_position_encoding_kwargs=dict(num_channels=perceiver_config.d_latents, index_dims=1),
                use_query_residual=True)

model=PerceiverModel(config=perceiver_config,decoder=decoder)
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)