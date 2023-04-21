import torch
from perceiver_pytorch import PerceiverIO
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverClassificationDecoder,
)

class PerceiverModel(nn.Module):

    def __init__(self,dim,
                queries_dim,num_classes,depth,num_latents,cross_heads,
                latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, use_queries=False):

        super().__init__()

        """
        Inputs:
            dim - Dimensionality of the input
            queries_dim - Dimensionality of the queries
            num_classes - Number of classes to predict
            depth - Depth of the Perceiver model
            num_latents - Number of latents, or induced set points, or centroids.
            cross_heads - Number of heads for cross attention.
            latent_heads - Number of heads for latent self attention.
            cross_dim_head - Dimensionality of each cross attention head.
            latent_dim_head - Dimensionality of each latent self attention head.
            latent_dim - Dimensionality of latent vectors
            weight_tie_layers - Whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob - Dropout probability for the cross attention probabilities
            use_queries - Whether to use queries
        """

        #basic parameters
        self.dim=dim
        self.queries_dim=queries_dim
        self.logits_dim=num_classes 
        self.depth=depth
        self.num_latents=num_latents
        self.cross_heads=cross_heads
        self.latent_heads=latent_heads
        self.cross_dim_head=cross_dim_head
        self.latent_dim_head=latent_dim_head
        self.latent_dim=latent_dim
        self.weight_tie_layers=weight_tie_layers
        self.seq_dropout_prob=seq_dropout_prob
        self.use_queries=use_queries

        #initialize perceiver IO model
        self.perceiver_model=PerceiverIO(
            dim=self.dim,
            queries_dim=self.queries_dim,
            logits_dim=self.logits_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=self.weight_tie_layers,
            seq_dropout_prob=self.seq_dropout_prob)

        #check if you need the queries
        if self.use_queries is False:
            self.classifier=nn.Linear(self.latent_dim,self.logits_dim) #needed when we do not pass the queries inside


    def forward(self,inputs, mask, queries=None):

        #check if you need the queries
        if self.use_queries is False:
            #get the latent vectors
            latent_vectors=self.perceiver_model(inputs,mask=mask)

            #perform mean pooling in terms of the sequence length
            latent_vectors=latent_vectors.mean(dim=1)
            #get the logits
            logits=self.classifier(latent_vectors)
            #return the logits
            return logits

        else:
            #get the logits
            logits=self.perceiver_model(inputs,queries)
            #return the logits
            return logits
        
class Perceiver_AudioVisual_Model(nn.Module):

    def __init__(self,audio_dim,video_dim,dim,
                queries_dim,num_classes,depth,num_latents,cross_heads,
                latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, use_queries=False):

        super().__init__()

        """
        Inputs:
            audio_dim - Dimensionality of the audio input
            video_dim - Dimensionality of the video input
            dim - Dimensionality of the input
            queries_dim - Dimensionality of the queries
            num_classes - Number of classes to predict
            depth - Depth of the Perceiver model
            num_latents - Number of latents, or induced set points, or centroids.
            cross_heads - Number of heads for cross attention.
            latent_heads - Number of heads for latent self attention.
            cross_dim_head - Dimensionality of each cross attention head.
            latent_dim_head - Dimensionality of each latent self attention head.
            latent_dim - Dimensionality of latent vectors
            weight_tie_layers - Whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob - Dropout probability for the cross attention probabilities
            use_queries - Whether to use queries

        """

        #basic parameters
        self.dim=dim
        self.audio_dim=audio_dim #dimension of the audio modality 
        self.video_dim=video_dim #dimension of the video modality
        self.queries_dim=queries_dim #dimension of the queries 
        self.logits_dim=num_classes #number of classes
        self.depth=depth #depth of the perceiver model
        self.num_latents=num_latents #number of latent vectors
        self.cross_heads=cross_heads #number of cross attention heads 
        self.latent_heads=latent_heads #number of latent attention heads
        self.cross_dim_head=cross_dim_head #dimension of the cross dimension attention heads
        self.latent_dim_head=latent_dim_head #dimension of the latent attention heads
        self.latent_dim=latent_dim #dimension of the latent vectors
        self.weight_tie_layers=weight_tie_layers #weight tie layers
        self.seq_dropout_prob=seq_dropout_prob #sequence dropout probability
        self.use_queries=use_queries #use queries or not

        #initialize linear layer 
        if(self.dim!=self.audio_dim):

            #map the audio to the same dimension as the video
            self.audio_linear=nn.Linear(self.audio_dim,self.dim)

        if(self.dim!=self.video_dim):

            #map the video to the same dimension as the audio
            self.video_linear=nn.Linear(self.video_dim,self.dim)

        #initialize perceiver IO model
        self.perceiver_model=PerceiverIO(
            dim=self.dim,
            queries_dim=self.queries_dim,
            logits_dim=self.logits_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=self.weight_tie_layers,
            seq_dropout_prob=self.seq_dropout_prob)
        
        #check if you need the queries
        if self.use_queries is False:
            self.classifier=nn.Linear(self.latent_dim,self.logits_dim) #needed when we do not pass the queries inside


    def forward(self,audio_inputs,visual_inputs,audio_mask,visual_mask,queries=None):
        
        if(self.dim!=self.audio_dim):
            #map the audio to the same dimension as the video
            audio_inputs=self.audio_linear(audio_inputs)

        if(self.dim!=self.video_dim):
            #map the video to the same dimension as the audio
            visual_inputs=self.video_linear(visual_inputs)

        #input embeddings concatenate 
        inputs=torch.cat((audio_inputs,visual_inputs),dim=1)

        #input mask concatenate
        mask=torch.cat((audio_mask,visual_mask),dim=1)

        #check if you need the queries
        if self.use_queries is False:
            latent_vectors=self.perceiver_model(inputs,mask=mask)
            #perform mean pooling in terms of the sequence length
            latent_vectors=latent_vectors.mean(dim=1)
            #get the logits
            logits=self.classifier(latent_vectors)
            #return the logits
            return logits
        
        else:
            #get the logits
            logits=self.perceiver_model(inputs,queries)
            #return the logits
            return logits


class Perceiver_TextVisual_model(nn.Module):

    def __init__(self,text_dim,video_dim,dim,bert_model_name,
                queries_dim,num_classes,depth,num_latents,cross_heads,
                latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, use_queries=False):

        super().__init__()

        """
        Inputs:
            text_dim - Dimensionality of the text input
            video_dim - Dimensionality of the video input
            dim - Dimensionality of the input
            bert_model_name - Name of the bert model
            queries_dim - Dimensionality of the queries
            num_classes - Number of classes to predict
            depth - Depth of the Perceiver model
            num_latents - Number of latents, or induced set points, or centroids.
            cross_heads - Number of heads for cross attention.
            latent_heads - Number of heads for latent self attention.
            cross_dim_head - Dimensionality of each cross attention head.
            latent_dim_head - Dimensionality of each latent self attention head.
            latent_dim - Dimensionality of latent vectors
            weight_tie_layers - Whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob - Dropout probability for the cross attention probabilities
            use_queries - Whether to use queries
        """


        #basic parameters
        self.dim=dim
        self.text_dim=text_dim #dimension of the audio modality
        self.video_dim=video_dim #dimension of the video modality
        self.bert_model_name=bert_model_name #bert model
        self.queries_dim=queries_dim #dimension of the queries
        self.logits_dim=num_classes #number of classes
        self.depth=depth #depth of the perceiver model
        self.num_latents=num_latents #number of latent vectors
        self.cross_heads=cross_heads #number of cross attention heads
        self.latent_heads=latent_heads #number of latent attention heads
        self.cross_dim_head=cross_dim_head #dimension of the cross dimension attention heads
        self.latent_dim_head=latent_dim_head #dimension of the latent attention heads
        self.latent_dim=latent_dim #dimension of the latent vectors
        self.weight_tie_layers=weight_tie_layers #weight tie layers
        self.seq_dropout_prob=seq_dropout_prob #sequence dropout probability
        self.use_queries=use_queries #use queries or not

        #initialize bert model
        self.bert_model=BertModel.from_pretrained(self.bert_model_name)

        #freeze the gradient
        for param in self.bert_model.parameters():
            param.requires_grad=False

        #initialize linear layer 
        if(self.dim!=self.text_dim):

            #map the audio to the same dimension as the video
            self.text_linear=nn.Linear(self.text_dim,self.dim)

        if(self.dim!=self.video_dim):

            #map the video to the same dimension as the audio
            self.video_linear=nn.Linear(self.video_dim,self.dim)

        #initialize perceiver IO model
        self.perceiver_model=PerceiverIO(
            dim=self.dim,
            queries_dim=self.queries_dim,
            logits_dim=self.logits_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=self.weight_tie_layers,
            seq_dropout_prob=self.seq_dropout_prob)
        
        #check if you need the queries
        if self.use_queries is False:
            self.classifier=nn.Linear(self.latent_dim,self.logits_dim) #needed when we do not pass the queries inside

    def forward(self,input_ids,visual_inputs,text_mask,visual_mask,queries=None):
        
        #bert output
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_inputs=bert_output[0]

        if(self.dim!=self.video_dim):
            #map the video to the same dimension as the audio
            visual_inputs=self.video_linear(visual_inputs)

        if(self.dim!=self.text_dim):
            #map the audio to the same dimension as the video
            text_inputs=self.text_linear(text_inputs)

        #input embeddings concatenate 
        inputs=torch.cat((text_inputs,visual_inputs),dim=1)

        text_mask=text_mask.bool()
        visual_mask=visual_mask.bool()

        mask=torch.cat((text_mask,visual_mask),dim=1)

        #check if you need the queries
        if self.use_queries is False:

            latent_vectors=self.perceiver_model(inputs,mask=mask)
            #perform mean pooling in terms of the sequence length
            latent_vectors=latent_vectors.mean(dim=1)
            #get the logits
            logits=self.classifier(latent_vectors)
            #return the logits
            return logits
        
        else:
            #get the logits
            logits=self.perceiver_model(inputs,queries)
            #return the logits
            return logits

class Perceiver_SBERT_TextVisual_model(nn.Module):

    def __init__(self,text_dim,video_dim,dim,queries_dim,num_classes,depth,num_latents,
                 cross_heads,latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, use_queries=False):

        super().__init__()

        """
            text_dim - Dimensionality of the text modality
            video_dim - Dimensionality of the video modality
            dim - Dimensionality of the input
            queries_dim - Dimensionality of the queries
            num_classes - Number of classes
            depth - Depth of the model
            num_latents - Number of latent vectors
            cross_heads - Number of cross attention heads
            latent_heads - Number of latent attention heads
            cross_dim_head - Dimensionality of each cross attention head
            latent_dim_head - Dimensionality of each latent attention head
            latent_dim - Dimensionality of the latent vectors
            weight_tie_layers - Whether to weight tie layers
            seq_dropout_prob - Sequence dropout probability
            use_queries - Whether to use queries or not

        """
        #basic parameters
        self.dim=dim
        self.text_dim=text_dim #dimension of the audio modality
        self.video_dim=video_dim #dimension of the video modality
        self.queries_dim=queries_dim #dimension of the queries
        self.logits_dim=num_classes #number of classes
        self.depth=depth #depth of the perceiver model
        self.num_latents=num_latents #number of latent vectors
        self.cross_heads=cross_heads #number of cross attention heads
        self.latent_heads=latent_heads #number of latent attention heads
        self.cross_dim_head=cross_dim_head #dimension of the cross dimension attention heads
        self.latent_dim_head=latent_dim_head #dimension of the latent attention heads
        self.latent_dim=latent_dim #dimension of the latent vectors
        self.weight_tie_layers=weight_tie_layers #weight tie layers
        self.seq_dropout_prob=seq_dropout_prob #sequence dropout probability
        self.use_queries=use_queries #use queries or not

        #initialize linear layer 
        if(self.dim!=self.text_dim):

            #map the audio to the same dimension as the video
            self.text_linear=nn.Linear(self.text_dim,self.dim)

        if(self.dim!=self.video_dim):

            #map the video to the same dimension as the audio
            self.video_linear=nn.Linear(self.video_dim,self.dim)

        #initialize perceiver IO model
        self.perceiver_model=PerceiverIO(
            dim=self.dim,
            queries_dim=self.queries_dim,
            logits_dim=self.logits_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=self.weight_tie_layers,
            seq_dropout_prob=self.seq_dropout_prob)
        
        #check if you need the queries
        if self.use_queries is False:
            self.classifier=nn.Linear(self.latent_dim,self.logits_dim) #needed when we do not pass the queries inside

    def forward(self,text_inputs,visual_inputs,text_mask,visual_mask,queries=None):
        
        if(self.dim!=self.video_dim):
            #map the video to the same dimension as the audio
            visual_inputs=self.video_linear(visual_inputs)

        if(self.dim!=self.text_dim):
            #map the audio to the same dimension as the video
            text_inputs=self.text_linear(text_inputs)

        #input embeddings concatenate 
        inputs=torch.cat((text_inputs,visual_inputs),dim=1)

        #print(text_inputs.shape,visual_inputs.shape)
        #text mask and visual mask
        text_mask=text_mask.bool()
        visual_mask=visual_mask.bool()

        mask=torch.cat((text_mask,visual_mask),dim=1)

        #check if you need the queries
        if self.use_queries is False:

            latent_vectors=self.perceiver_model(inputs,mask=mask)
            #perform mean pooling in terms of the sequence length
            latent_vectors=latent_vectors.mean(dim=1)
            #get the logits
            logits=self.classifier(latent_vectors)
            #return the logits
            return logits
        
        else:
            #get the logits
            logits=self.perceiver_model(inputs,queries)
            #return the logits
            return logits

class Perceiver_TextVisual_multi_task_model(nn.Module):

    def __init__(self,text_dim,video_dim,dim,text_model_name,
                queries_dim,depth,num_latents,cross_heads,
                latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, task_dict, use_queries=False):
        
        super().__init__()

        """
            text_dim - Dimensionality of the text modality
            video_dim - Dimensionality of the video modality
            dim - Dimensionality of the input
            bert_model_name - Name of the bert model
            queries_dim - Dimensionality of the queries 
            depth - Depth of the model
            num_latents - Number of latent vectors
            cross_heads - Number of cross attention heads
            latent_heads - Number of latent attention heads
            cross_dim_head - Dimensionality of each cross attention head
            latent_dim_head - Dimensionality of each latent attention head
            latent_dim - Dimensionality of the latent vectors
            weight_tie_layers - Whether to weight tie layers
            seq_dropout_prob - Sequence dropout probability
            task_dict - Dictionary with the tasks
            use_queries - Whether to use queries or not
        """

        self.text_dim=text_dim #dimension of the text modality
        self.video_dim=video_dim #dimension of the video modality
        self.dim=dim #dimension of the input
        self.text_model_name=text_model_name #name of the text model
        self.queries_dim=queries_dim #dimension of the queries
        self.depth=depth #depth of the perceiver model
        self.num_latents=num_latents #number of latent vectors
        self.cross_heads=cross_heads #number of cross attention heads
        self.latent_heads=latent_heads #number of latent attention heads
        self.cross_dim_head=cross_dim_head #dimension of the cross dimension attention heads
        self.latent_dim_head=latent_dim_head #dimension of the latent attention heads
        self.latent_dim=latent_dim #dimension of the latent vectors
        self.weight_tie_layers=weight_tie_layers #weight tie layers
        self.seq_dropout_prob=seq_dropout_prob #sequence dropout probability
        self.task_dict=task_dict #dictionary with the tasks
        self.use_queries=use_queries #use queries or not

        #placheholder for the logits dim
        self.logits_dim=self.dim


        #initialize bert model
        self.bert_model=BertModel.from_pretrained(self.text_model_name)

        #freeze the gradient
        for param in self.bert_model.parameters():
            param.requires_grad=False

        #initialize linear layer 
        if(self.dim!=self.text_dim):

            #map the audio to the same dimension as the video
            self.text_linear=nn.Linear(self.text_dim,self.dim)

        if(self.dim!=self.video_dim):

            #map the video to the same dimension as the audio
            self.video_linear=nn.Linear(self.video_dim,self.dim)

        
        #initialize perceiver IO model
        self.perceiver_model=PerceiverIO(
            dim=self.dim,
            queries_dim=self.queries_dim,
            logits_dim=self.logits_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=self.weight_tie_layers,
            seq_dropout_prob=self.seq_dropout_prob)
        
        #check if you need the queries
        if self.use_queries is False:
            self.classifier=nn.Linear(self.latent_dim,self.logits_dim) #needed when we do not pass the queries inside

        self.task_fc_dict=nn.ModuleDict()
        for task in self.task_dict.keys():
            self.task_fc_dict['fc_'+task]=nn.Linear(self.latent_dim, self.task_dict[task])

    def forward(self,input_ids,visual_inputs,text_mask,visual_mask,queries=None):
        
        #bert output
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_inputs=bert_output[0]

        if(self.dim!=self.video_dim):
            #map the video to the same dimension as the audio
            visual_inputs=self.video_linear(visual_inputs)

        if(self.dim!=self.text_dim):
            #map the audio to the same dimension as the video
            text_inputs=self.text_linear(text_inputs)

        #input embeddings concatenate 
        inputs=torch.cat((text_inputs,visual_inputs),dim=1)

        #text and visual mask
        text_mask=text_mask.bool()
        visual_mask=visual_mask.bool()

        #overall concatenated mask 
        mask=torch.cat((text_mask,visual_mask),dim=1)

        #task outputs
        task_outputs=dict()

        #check if you need the queries
        if self.use_queries is False:

            latent_vectors=self.perceiver_model(inputs,mask=mask)
            #perform mean pooling in terms of the sequence length
            latent_vectors=latent_vectors.mean(dim=1)
            #print(latent_vectors.shape)

            for task in self.task_dict.keys():
                task_outputs[task]=self.task_fc_dict['fc_'+task](latent_vectors)

        return task_outputs

class Perceiver_AudioText_model(nn.Module):

    def __init__(self,text_dim,audio_dim,dim,bert_model_name,
                queries_dim,num_classes,depth,num_latents,cross_heads,
                latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, use_queries=False):

        super().__init__()

        """
        Inputs:
            text_dim - Dimensionality of the text input
            video_dim - Dimensionality of the video input
            dim - Dimensionality of the input
            bert_model_name - Name of the bert model
            queries_dim - Dimensionality of the queries
            num_classes - Number of classes to predict
            depth - Depth of the Perceiver model
            num_latents - Number of latents, or induced set points, or centroids.
            cross_heads - Number of heads for cross attention.
            latent_heads - Number of heads for latent self attention.
            cross_dim_head - Dimensionality of each cross attention head.
            latent_dim_head - Dimensionality of each latent self attention head.
            latent_dim - Dimensionality of latent vectors
            weight_tie_layers - Whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob - Dropout probability for the cross attention probabilities
            use_queries - Whether to use queries
        """


        #basic parameters
        self.dim=dim
        self.text_dim=text_dim #dimension of the audio modality
        self.audio_dim=audio_dim #dimension of the video modality
        self.bert_model_name=bert_model_name #bert model
        self.queries_dim=queries_dim #dimension of the queries
        self.logits_dim=num_classes #number of classes
        self.depth=depth #depth of the perceiver model
        self.num_latents=num_latents #number of latent vectors
        self.cross_heads=cross_heads #number of cross attention heads
        self.latent_heads=latent_heads #number of latent attention heads
        self.cross_dim_head=cross_dim_head #dimension of the cross dimension attention heads
        self.latent_dim_head=latent_dim_head #dimension of the latent attention heads
        self.latent_dim=latent_dim #dimension of the latent vectors
        self.weight_tie_layers=weight_tie_layers #weight tie layers
        self.seq_dropout_prob=seq_dropout_prob #sequence dropout probability
        self.use_queries=use_queries #use queries or not

        #initialize bert model
        self.bert_model=BertModel.from_pretrained(self.bert_model_name)

        #freeze the gradient
        for param in self.bert_model.parameters():
            param.requires_grad=False

        #initialize linear layer 
        if(self.dim!=self.text_dim):

            self.text_linear=nn.Linear(self.text_dim,self.dim)

        if(self.dim!=self.audio_dim):

            self.audio_linear=nn.Linear(self.audio_dim,self.dim)

        #initialize perceiver IO model
        self.perceiver_model=PerceiverIO(
            dim=self.dim,
            queries_dim=self.queries_dim,
            logits_dim=self.logits_dim,
            depth=self.depth,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=self.weight_tie_layers,
            seq_dropout_prob=self.seq_dropout_prob)
        
        #check if you need the queries
        if self.use_queries is False:
            self.classifier=nn.Linear(self.latent_dim,self.logits_dim) #needed when we do not pass the queries inside

    def forward(self,input_ids,audio_inputs,text_mask,audio_mask,queries=None):
        
        #bert output
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_inputs=bert_output[0]

        if(self.dim!=self.audio_dim):
            #map the video to the same dimension as the audio
            audio_inputs=self.audio_linear(audio_inputs)

        if(self.dim!=self.text_dim):
            #map the audio to the same dimension as the video
            text_inputs=self.text_linear(text_inputs)

        #input embeddings concatenate 
        inputs=torch.cat((text_inputs,audio_inputs),dim=1)

        text_mask=text_mask.bool()
        audio_mask=audio_mask.bool()

        mask=torch.cat((text_mask,audio_mask),dim=1)

        #check if you need the queries
        if self.use_queries is False:

            latent_vectors=self.perceiver_model(inputs,mask=mask)
            #perform mean pooling in terms of the sequence length
            latent_vectors=latent_vectors.mean(dim=1)
            #get the logits
            logits=self.classifier(latent_vectors)
            #return the logits
            return logits
        
        else:
            #get the logits
            logits=self.perceiver_model(inputs,queries)
            #return the logits
            return logits





#d_model is a concatenation of the audio and video dimensions (project everything to the same dimension and input dimension for the perceiver is audio + video dimensions)
# class Perceiver_TextVisual_hf_model(nn.Module):
#     def __init__(self,d_model,d_latents,num_labels,num_self_attends_per_block,num_classes):
        
#         super().__init__()

#         self.d_model=d_model
#         self.d_latents=d_latents
#         self.num_labels=num_labels
#         self.num_self_attends_per_block=num_self_attends_per_block
#         self.num_classes=num_classes

#         self.perceiver_config = PerceiverConfig(d_model=self.d_model,num_labels=num_classes)

#         self.decoder=PerceiverClassificationDecoder(self.perceiver_config,
#                 num_channels=self.perceiver_config.d_latents,
#                 trainable_position_encoding_kwargs=dict(num_channels=self.perceiver_config.d_latents, index_dims=1),
#                 use_query_residual=True)

#         self.model=PerceiverModel(self.perceiver_config,decoder=self.decoder)

#     def forward(self,inputs,mask,queries=None):
#         #basic parameters

#         #get the logits
#         logits=self.model(inputs,mask=mask,queries=queries)

#         #return the logits
#         return logits
    

# d_model=256
# d_latents=128
# num_labels=10
# num_self_attends_per_block=4
# num_classes=10

# perceiver_config = PerceiverConfig(d_model=d_model,num_labels=num_classes, num_self_attends_per_block=num_self_attends_per_block,num_cross_attention_heads=4)
# print(perceiver_config)
# decoder=PerceiverClassificationDecoder(perceiver_config,
#                 num_channels=perceiver_config.d_latents,
#                 trainable_position_encoding_kwargs=dict(num_channels=perceiver_config.d_latents, index_dims=1),
#                 use_query_residual=True)

# model=PerceiverModel(config=perceiver_config,decoder=decoder)
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)
# if __name__=="__main__":

#     #define the parameters
#     dim=512
#     queries_dim=512
#     num_classes=10
#     depth=6
#     num_latents=512
#     cross_heads=1
#     latent_heads=8
#     cross_dim_head=64
#     latent_dim_head=64
#     latent_dim=512
#     weight_tie_layers=True
#     seq_dropout_prob=0.1
#     use_queries=False
#     audio_dim=768
#     video_dim=512

#     #create the model
#     model=Perceiver_AudioVisual_Model(audio_dim,video_dim,dim,
#                         queries_dim,
#                         num_classes,depth,num_latents,cross_heads,
#                         latent_heads,cross_dim_head,
#                         latent_dim_head,latent_dim,
#                         weight_tie_layers,seq_dropout_prob,use_queries)

#     #print the model
#     #print(model)


#     #compute model parameters 
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     params = sum([np.prod(p.size()) for p in model_parameters])
#     print('Number of parameters: %d' %(params))
#     model=model.to(device)

#     #define the inputs
#     audio_inputs=torch.randn(1,50,768)
#     visual_inputs=torch.randn(1,50,512)
#     audio_mask=torch.ones(1,50)
#     visual_mask=torch.ones(1,50)
#     queries=torch.randn(1,1,512)

#     audio_inputs=audio_inputs.to(device)
#     visual_inputs=visual_inputs.to(device)
#     audio_mask=audio_mask.to(device)
#     visual_mask=visual_mask.to(device)

#     #convert mask to boolean
#     audio_mask=audio_mask.bool()
#     visual_mask=visual_mask.bool()


#     #get the logits
#     logits=model(audio_inputs=audio_inputs,
#                 visual_inputs=visual_inputs,
#                 audio_mask=audio_mask,
#                 visual_mask=visual_mask,
#                 queries=queries)

#     #print the logits
#     print(logits.shape)

    