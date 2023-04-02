import torch
from perceiver_pytorch import PerceiverIO
import torch.nn as nn
import torch.nn.functional as F

class PerceiverModel(nn.Module):

    def __init__(self,dim,
                queries_dim,num_classes,depth,num_latents,cross_heads,
                latent_heads,cross_dim_head,latent_dim_head,latent_dim,
                weight_tie_layers,seq_dropout_prob, use_queries=False):

        super().__init__()

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

        #basic parameters
        self.dim=dim
        self.audio_dim=audio_dim
        self.video_dim=video_dim
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

        #initialize linear layer 
        
        if(self.dim!=self.audio_dim):

            #map the audio to the same dimension as the video
            self.audio_linear=nn.Linear(self.audio_dim,self.dim)

        elif(self.dim!=self.video_dim):

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


    def forward(self, audio_inputs,visual_inputs,audio_mask,visual_mask,queries=None):
        
        if(self.dim!=self.audio_dim):
            #map the audio to the same dimension as the video
            audio_inputs=self.audio_linear(audio_inputs)

        elif(self.dim!=self.video_dim):
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


if __name__=="__main__":

    #define the parameters
    dim=512
    queries_dim=512
    num_classes=10
    depth=6
    num_latents=512
    cross_heads=1
    latent_heads=8
    cross_dim_head=64
    latent_dim_head=64
    latent_dim=512
    weight_tie_layers=True
    seq_dropout_prob=0.1
    use_queries=False
    audio_dim=768
    video_dim=512

    #create the model
    model=Perceiver_AudioVisual_Model(audio_dim,video_dim,dim,
                        queries_dim,
                        num_classes,depth,num_latents,cross_heads,
                        latent_heads,cross_dim_head,
                        latent_dim_head,latent_dim,
                        weight_tie_layers,seq_dropout_prob,use_queries)

    #print the model
    #print(model)

    #define the inputs
    audio_inputs=torch.randn(1,50,768)
    visual_inputs=torch.randn(1,50,512)
    audio_mask=torch.ones(1,50)
    visual_mask=torch.ones(1,50)
    queries=torch.randn(1,1,512)

    #convert mask to boolean
    audio_mask=audio_mask.bool()
    visual_mask=visual_mask.bool()


    #get the logits
    logits=model(audio_inputs=audio_inputs,
                visual_inputs=visual_inputs,
                audio_mask=audio_mask,
                visual_mask=visual_mask,
                queries=queries)

    #print the logits
    print(logits.shape)

    