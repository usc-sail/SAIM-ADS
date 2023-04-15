from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os 
import torch 
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import pickle
import json 

####### dataset declaration for binary tone classification using LSTM #######
class SAIM_ads_tone_clip_features_dataset(Dataset):
    def __init__(self,csv_data,label_map,num_classes,max_length,fps,base_fps):  

        self.csv_data=csv_data
        self.num_classes=num_classes
        self.max_length=max_length
        self.fps=fps
        self.base_fps=base_fps
        self.division_factor=self.base_fps//self.fps # 24/4=6 
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        self.label_map=label_map

    def __len__(self):
        return(len(self.clip_feature_list))

    def subsample_feature(self,feature_array):
        
        #feature_array - subsample the array by extracting the features at every frame at self.division_factor

        feature_array_subsampled=feature_array[::self.division_factor]
        #print(feature_array_subsampled.shape,feature_array.shape)
        return(feature_array_subsampled)

    def pad_data(self,feat_data):
        padded=np.zeros((self.max_length,feat_data.shape[1]))
        if(feat_data.shape[0]>self.max_length):
            padded=feat_data[:self.max_length,:]
        else:
            padded[:feat_data.shape[0],:]=feat_data
        return(padded)

    def __getitem__(self,idx):

        #get path of the feature fiile 
        clip_feature_file=self.clip_feature_list[idx]

        #load the feature file
        with open(clip_feature_file, 'rb') as f:
            clip_features = pickle.load(f)

        #get the features
        clip_feature_array=clip_features['Features']

        #subsample the features
        clip_feature_array_subsampled=self.subsample_feature(clip_feature_array)

        #return the length of the features
        #print(clip_feature_array_subsampled.shape)
        if(clip_feature_array_subsampled.shape[0]>=self.max_length):
            feat_len=self.max_length
        else:
            feat_len=clip_feature_array_subsampled.shape[0]

        #pad the features
        clip_feature_array_padded=self.pad_data(clip_feature_array_subsampled)

        #get the label
        transition_tone=self.label_map[self.csv_data['Transition_val'].iloc[idx]]

        transition_label=np.zeros((self.num_classes))
        transition_label[transition_tone]=1


        return(clip_feature_array_padded,transition_label,feat_len)

####### dataset declaration for social message absence/presence classification using LSTM #######

class SAIM_social_message_clip_features_dataset(Dataset):
    def __init__(self,csv_data,label_map,num_classes,max_length,fps,base_fps):  

        self.csv_data=csv_data
        self.num_classes=num_classes
        self.max_length=max_length
        self.fps=fps
        self.base_fps=base_fps
        self.division_factor=self.base_fps//self.fps # 24/4=6 
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        self.label_map=label_map

    def __len__(self):
        return(len(self.clip_feature_list))

    def subsample_feature(self,feature_array):
        
        #feature_array - subsample the array by extracting the features at every frame at self.division_factor
        feature_array_subsampled=feature_array[::self.division_factor]
        #print(feature_array_subsampled.shape,feature_array.shape)
        return(feature_array_subsampled)

    def pad_data(self,feat_data):
        padded=np.zeros((self.max_length,feat_data.shape[1]))
        if(feat_data.shape[0]>self.max_length):
            padded=feat_data[:self.max_length,:]
        else:
            padded[:feat_data.shape[0],:]=feat_data
        return(padded)

    def __getitem__(self,idx):

        #get path of the feature fiile 
        clip_feature_file=self.clip_feature_list[idx]

        #load the feature file
        with open(clip_feature_file, 'rb') as f:
            clip_features = pickle.load(f)

        #get the features
        clip_feature_array=clip_features['Features']

        #subsample the features
        clip_feature_array_subsampled=self.subsample_feature(clip_feature_array)

        #return the length of the features
        #print(clip_feature_array_subsampled.shape)
        if(clip_feature_array_subsampled.shape[0]>=self.max_length):
            feat_len=self.max_length
        else:
            feat_len=clip_feature_array_subsampled.shape[0]

        #pad the features
        clip_feature_array_padded=self.pad_data(clip_feature_array_subsampled)

        #get the label
        social_msg=self.label_map[self.csv_data['social_message'].iloc[idx]]

        social_msg_label=np.zeros((self.num_classes))
        social_msg_label[social_msg]=1


        return(clip_feature_array_padded,social_msg_label,feat_len)

####### dataset declaration for single task using MHA dataset #######
class SAIM_single_task_dataset(Dataset):  # tobe integrated later 
    def __init__(self,csv_data,label_map,num_classes,max_length,fps,base_fps,task_name):  

        self.csv_data=csv_data
        self.num_classes=num_classes
        self.max_length=max_length
        self.fps=fps
        self.base_fps=base_fps
        self.division_factor=self.base_fps//self.fps # 24/4=6 
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        self.label_map=label_map
        self.task_name=task_name
        

    def __len__(self):
        return(len(self.clip_feature_list))

    def subsample_feature(self,feature_array):
        
        #feature_array - subsample the array by extracting the features at every frame at self.division_factor
        feature_array_subsampled=feature_array[::self.division_factor]
        return(feature_array_subsampled)

    def pad_data(self,feat_data):

        padded=np.zeros((self.max_length,feat_data.shape[1]))
        
        if(feat_data.shape[0]>self.max_length):
            padded=feat_data[:self.max_length,:]
            attn_mask=np.ones((self.max_length))
        else:
            attn_mask=np.zeros((self.max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)

    def __getitem__(self,idx):

        #get path of the feature fiile 
        clip_feature_file=self.clip_feature_list[idx]

        #load the feature file
        with open(clip_feature_file, 'rb') as f:
            clip_features = pickle.load(f)

        #get the features
        clip_feature_array=clip_features['Features']

        #subsample the features
        clip_feature_array_subsampled=self.subsample_feature(clip_feature_array)

        #pad the features
        clip_feature_array_padded,attention_mask=self.pad_data(clip_feature_array_subsampled)

        #get the label
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):

            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]


        return(clip_feature_array_padded,ret_label,attention_mask)

### dataset for shot level modeling single task ###
class SAIM_single_task_dataset_shot_level(Dataset):

    def __init__(self,csv_data,base_folder,label_map,num_classes,max_length,task_name):  

        #arguments here 
        self.csv_data=csv_data
        self.num_classes=num_classes
        self.max_length=max_length
        self.base_folder=base_folder
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()

        #shot feature list
        self.shot_feature_list=[os.path.join(self.base_folder,file.split("/")[-1]) for file in self.clip_feature_list]

        #label map and task names
        self.label_map=label_map
        self.task_name=task_name

    def __len__(self):
        return(len(self.csv_data))
    
    def pad_data(self,feat_data):

        padded=np.zeros((self.max_length,feat_data.shape[1]))
        
        if(feat_data.shape[0]>self.max_length):
            padded=feat_data[:self.max_length,:]
            attn_mask=np.ones((self.max_length))
        else:
            attn_mask=np.zeros((self.max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)
    
    def __getitem__(self,idx):

        filename=self.shot_feature_list[idx]
        #print(filename)

        #load the feature file
        try:
            with open(filename, 'rb') as f:
                shot_features = pickle.load(f)
        except:
            print(filename)

        #get the keys 
        keys=sorted(list(shot_features.keys()))

        #get the features
        shot_feature_avg=[]
        for key in keys:
            shot_feat_temp=shot_features[key]
            #print(shot_feat_temp.shape)
            if(len(shot_feat_temp)>0):
                shot_feat_avg=np.mean(shot_feat_temp,axis=0)
                shot_feature_avg.append(shot_feat_avg)

        #convert to numpy array
        shot_feature_avg=np.array(shot_feature_avg)
        if(len(shot_feature_avg)==0):
            print(filename)

        #shot features and attention mask
        shot_feat_padded,attention_mask=self.pad_data(shot_feature_avg)

        #get the label
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):

            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]

        #return the shot features, return label and attention mask
        return(shot_feat_padded,ret_label,attention_mask)

### dataset for audio only modeling single task ###
class SAIM_single_task_dataset_audio_only(Dataset): #audio only dataset

    def __init__(self,csv_data,embedding_file,label_map,num_classes,max_length,task_name):
         
        #arguments here
        self.csv_data=csv_data
        self.num_classes=num_classes
        self.max_length=max_length
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        self.label_map=label_map
        self.task_name=task_name
        self.embedding_file=embedding_file
        #print(self.task_name)

        #load the embedding file
        with open(self.embedding_file, 'rb') as f:
            self.embedding = pickle.load(f)

        #ast embeddings
        self.ast_embeds=self.embedding['data']['embeddings']

        #get the keys
        self.clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in self.clip_feature_list]

    def __len__(self):
        #csv data
        return(len(self.csv_data))
    
    def pad_data(self,feat_data):

        #padded data and attention mask 
        padded=np.zeros((self.max_length,feat_data.shape[1]))
        
        if(feat_data.shape[0]>self.max_length):
            padded=feat_data[:self.max_length,:]
            attn_mask=np.ones((self.max_length))
        else:
            attn_mask=np.zeros((self.max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)
    
    def __getitem__(self,idx):

        #get the clip key
        clip_key=self.clip_keys[idx]

        #get the audio features
        audio_feat=self.ast_embeds[clip_key].cpu().numpy()

        #pad the data
        audio_feat_padded,attention_mask=self.pad_data(audio_feat)

        #get the label
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):

            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]

        #return the shot features, return label and attention mask
        return(audio_feat_padded,ret_label,attention_mask)

## dataset for audio and video modeling single task (to be used with Perceiver) ###
class SAIM_single_task_dataset_audio_visual(Dataset):

    def __init__(self,csv_data,embedding_file,label_map,num_classes,audio_max_length,video_max_length,task_name):
         
        #arguments here
        self.csv_data=csv_data
        self.num_classes=num_classes

        #audio and video max length
        self.audio_max_length=audio_max_length
        self.video_max_length=video_max_length

        #clip feature list
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        self.label_map=label_map
        self.task_name=task_name
        self.embedding_file=embedding_file

        #load the embedding file
        with open(self.embedding_file, 'rb') as f:
            self.embedding = pickle.load(f)

        #ast embeddings
        self.ast_embeds=self.embedding['data']['embeddings']

        #get the keys
        self.clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in self.clip_feature_list]

    ### length of the csv data ###
    def __len__(self):
        #csv data
        return(len(self.csv_data))
    
    ### pad the data ###
    def pad_data(self,feat_data,max_length):

        #padded data and attention mask 
        padded=np.zeros((max_length,feat_data.shape[1]))
        
        ### 
        if(feat_data.shape[0]>max_length):
            padded=feat_data[:max_length,:]
            attn_mask=np.ones((max_length))
        else:
            attn_mask=np.zeros((max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)
    
    def __getitem__(self,idx):

        #get the clip key
        clip_key=self.clip_keys[idx]

        #get the clip feature path
        clip_feat_path=self.clip_feature_list[idx]
    
        #get the audio features
        audio_feat=self.ast_embeds[clip_key].cpu().numpy()

        #get the video features
        with open(clip_feat_path, 'rb') as f:
            video_feat = pickle.load(f)

        #pad the data
        audio_feat_padded,attention_mask_audio=self.pad_data(audio_feat,self.audio_max_length) #audio feature padded to maximum length
        video_feat_padded,attention_mask_video=self.pad_data(video_feat['Features'],self.video_max_length) #video feature padded to maximum length

        #get the label
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):

            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]

        #return the audio, video features + attention mask and return label
        return(audio_feat_padded,
                video_feat_padded,
                ret_label,
                attention_mask_audio,attention_mask_video)
    
class SAIM_single_task_dataset_visual_text(Dataset):

    def __init__(self,csv_data,transcripts_file,tokenizer,
                    label_map,num_classes,text_max_length,video_max_length,task_name):

        #arguments here
        self.csv_data=csv_data
        self.num_classes=num_classes
        self.transcripts_file=transcripts_file
        self.label_map=label_map
        self.task_name=task_name
        self.tokenizer=tokenizer

        with open(self.transcripts_file, 'r') as f:
            self.transcripts_dict = json.load(f)

        #audio and video max length
        self.text_max_length=text_max_length
        self.video_max_length=video_max_length

        #clip feature list
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        
        #get the keys
        self.clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in self.clip_feature_list]

    ### length of the csv data ###
    def __len__(self):
        #csv data
        return(len(self.csv_data))
    
    ### pad the data ###
    def pad_data(self,feat_data,max_length):

        #padded data and attention mask 
        padded=np.zeros((max_length,feat_data.shape[1]))
        
        ### 
        if(feat_data.shape[0]>max_length):
            padded=feat_data[:max_length,:]
            attn_mask=np.ones((max_length))
        else:
            attn_mask=np.zeros((max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)

    def __getitem__(self,idx):


        #get the clip key
        clip_key=self.clip_keys[idx]

        #get the clip feature path
        clip_feat_path=self.clip_feature_list[idx]

        #get the video features
        with open(clip_feat_path, 'rb') as f:
            video_feat = pickle.load(f)

        #padded video features 
        video_feat_padded,attention_mask_video=self.pad_data(video_feat['Features'],self.video_max_length) #video feature padded to maximum length

        if(clip_key not in self.transcripts_dict):

            #empty transcripts or [MASK] token transcripts 
            #print(clip_key)
            # create input ids  for text
            input_ids = torch.tensor([[101] + [103] * (self.text_max_length - 2) + [102]])
            # create attention mask  for text
            attn_mask = torch.zeros(self.text_max_length, dtype=torch.long)

            #create token type ids
            token_type_ids = torch.ones(self.text_max_length, dtype=torch.long)

        else:
            
            transcript=self.transcripts_dict[clip_key]
            #encode the caption
            encoded = self.tokenizer.encode_plus(
                text=transcript,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = self.text_max_length,  # maximum length of a sentence
                truncation=True,
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )

            # Get the input IDs and attention mask in tensor format
            input_ids = encoded['input_ids']
            attn_mask = encoded['attention_mask']
            token_type_ids = encoded['token_type_ids']

        #get the label
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):

            lb=self.csv_data[self.task_name].iloc[idx]
            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]

        
        return_dict={'input_ids':input_ids.squeeze(0),
                     'attention_mask':attn_mask.squeeze(0),
                     'token_type_ids':token_type_ids.squeeze(0),
                     'video_feat':video_feat_padded,
                     'video_attn_mask':attention_mask_video,
                     'label':ret_label}
        
        return(return_dict)

class SAIM_single_task_dataset_visual_text_shot_level(Dataset):

    def __init__(self,csv_data,transcripts_file,tokenizer,base_folder,
                    label_map,num_classes,text_max_length,video_max_length,task_name):
        #arguments here
        self.csv_data=csv_data
        self.num_classes=num_classes
        self.transcripts_file=transcripts_file
        self.label_map=label_map
        self.task_name=task_name
        self.tokenizer=tokenizer
        self.base_folder=base_folder
        with open(self.transcripts_file, 'r') as f:
            self.transcripts_dict = json.load(f)

        #audio and video max length
        self.text_max_length=text_max_length
        self.video_max_length=video_max_length

        #clip feature list
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()

        #shot feature list
        self.shot_feature_list=[os.path.join(self.base_folder,file.split("/")[-1]) for file in self.clip_feature_list]
        
        #get the keys
        self.clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in self.clip_feature_list]

    def __len__(self):
        return(len(self.csv_data))
    
    def pad_data(self,feat_data,max_length):

        padded=np.zeros((max_length,feat_data.shape[1]))
        
        if(feat_data.shape[0]>max_length):
            padded=feat_data[:max_length,:]
            attn_mask=np.ones((max_length))
        else:
            attn_mask=np.zeros((max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)
    
    def __getitem__(self,idx):

        #get the clip key
        clip_key=self.clip_keys[idx]
        filename=self.shot_feature_list[idx]
        #print(filename,clip_key)

        #load the feature file
        try:
            with open(filename, 'rb') as f:
                shot_features = pickle.load(f)
        except:
            print(filename)

        #get the keys 
        keys=sorted(list(shot_features.keys()))

        #get the features
        shot_feature_avg=[]
        for key in keys:
            shot_feat_temp=shot_features[key]
            #print(shot_feat_temp.shape)
            if(len(shot_feat_temp)>0):
                shot_feat_avg=np.mean(shot_feat_temp,axis=0)
                shot_feature_avg.append(shot_feat_avg)

        #convert to numpy array
        shot_feature_avg=np.array(shot_feature_avg)
        if(len(shot_feature_avg)==0):
            print(filename)

        #shot features and attention mask
        shot_feat_padded,attention_mask=self.pad_data(shot_feature_avg,self.video_max_length)

        #get the label here 
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):
            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]

        #clip key and transcripts parsing here 
        if(clip_key not in self.transcripts_dict):

            #empty transcripts or [MASK] token transcripts 
            # create input ids  for text
            input_ids = torch.tensor([[101] + [103] * (self.text_max_length - 2) + [102]])
            # create attention mask  for text
            attn_mask = torch.zeros(self.text_max_length, dtype=torch.long)

            #create token type ids
            token_type_ids = torch.ones(self.text_max_length, dtype=torch.long)

        else:
            
            transcript=self.transcripts_dict[clip_key]
            #encode the caption
            encoded = self.tokenizer.encode_plus(
                text=transcript,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = self.text_max_length,  # maximum length of a sentence
                truncation=True,
                padding='max_length',  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )
            # Get the input IDs and attention mask in tensor format
            input_ids = encoded['input_ids']
            attn_mask = encoded['attention_mask']
            token_type_ids = encoded['token_type_ids']

        return_dict={'input_ids':input_ids.squeeze(0),
                     'attention_mask':attn_mask.squeeze(0),
                     'token_type_ids':token_type_ids.squeeze(0),
                     'video_feat':shot_feat_padded,
                     'video_attn_mask':attention_mask,
                     'label':ret_label}

        return(return_dict)


class SAIM_single_task_dataset_visual_SBERT_text_shot_level(Dataset):

    def __init__(self,csv_data,transcripts_feat_file,base_folder,
                    label_map,num_classes,text_max_length,video_max_length,text_dim,task_name):
        #arguments here
        self.csv_data=csv_data
        self.num_classes=num_classes
        self.transcripts_feat_file=transcripts_feat_file
        self.label_map=label_map
        self.task_name=task_name
        self.base_folder=base_folder
        self.text_dim=text_dim

        #print(self.transcripts_feat_file)

        with open(self.transcripts_feat_file, 'rb') as f:
            self.transcripts_feat=pickle.load(f)

        #audio and video max length
        self.text_max_length=text_max_length
        self.video_max_length=video_max_length

        #clip feature list
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()

        #shot feature list
        self.shot_feature_list=[os.path.join(self.base_folder,file.split("/")[-1]) for file in self.clip_feature_list]
        
        #get the keys
        self.clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in self.clip_feature_list]

    def __len__(self):
        return(len(self.csv_data))
    
    def pad_data(self,feat_data,max_length):

        padded=np.zeros((max_length,feat_data.shape[1]))
        
        if(feat_data.shape[0]>max_length):
            padded=feat_data[:max_length,:]
            attn_mask=np.ones((max_length))
        else:
            attn_mask=np.zeros((max_length))
            padded[:feat_data.shape[0],:]=feat_data
            attn_mask[:feat_data.shape[0]]=1

        return(padded,attn_mask)
    
    def __getitem__(self,idx):

        #get the clip key
        clip_key=self.clip_keys[idx]
        filename=self.shot_feature_list[idx]

        #load the feature file
        try:
            with open(filename, 'rb') as f:
                shot_features = pickle.load(f)
        except:
            print(filename)

        #get the keys 
        keys=sorted(list(shot_features.keys()))

        #get the features
        shot_feature_avg=[]
        for key in keys:
            shot_feat_temp=shot_features[key]
            #print(shot_feat_temp.shape)
            if(len(shot_feat_temp)>0):
                shot_feat_avg=np.mean(shot_feat_temp,axis=0)
                shot_feature_avg.append(shot_feat_avg)

        #convert to numpy array
        shot_feature_avg=np.array(shot_feature_avg)
        
        #shot features and attention mask
        shot_feat_padded,attention_mask=self.pad_data(shot_feature_avg,self.video_max_length)

        #get the label here 
        if((self.task_name=='social_message') or (self.task_name=='Transition_val')):
            label_c=self.label_map[self.csv_data[self.task_name].iloc[idx]]
            ret_label=np.zeros((self.num_classes))
            ret_label[label_c]=1

        elif(self.task_name=='Topic'):
            ret_label=self.label_map[self.csv_data[self.task_name].iloc[idx]]

        #clip key and transcripts parsing here 
        if(clip_key not in self.transcripts_feat):

            text_feat_padded=np.zeros((self.text_max_length,self.text_dim))
            attn_mask=np.zeros((self.text_max_length))
            
        else:
            
            text_feat=self.transcripts_feat[clip_key]
            text_feat_padded,attn_mask=self.pad_data(text_feat,self.text_max_length)

        #print(text_feat_padded.shape,shot_feat_padded.shape)

        return_dict={'text_feat':text_feat_padded,
                     'attention_mask':attn_mask,
                     'video_feat':shot_feat_padded,
                     'video_attn_mask':attention_mask,
                     'label':ret_label}

        return(return_dict)


        
# csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
# csv_data=pd.read_csv(csv_file)
# transcripts_file="/data/digbose92/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.json"
# tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
# label_map={'No':0,'Yes':1}
# num_classes=2
# text_max_length=512
# video_max_length=333
# task_name='social_message'
# bert_model_name='bert-base-uncased'

# dataset=SAIM_single_task_dataset_visual_text(csv_data=csv_data,
#                                             transcripts_file=transcripts_file,
#                                             tokenizer=tokenizer,
#                                             label_map=label_map,
#                                             num_classes=num_classes,
#                                             text_max_length=text_max_length,
#                                             video_max_length=video_max_length,
#                                             task_name=task_name)
# dl=DataLoader(dataset,batch_size=2,shuffle=True)

# return_dict=next(iter(dl))

# bert_model=BertModel.from_pretrained(bert_model_name)
# bert_model.eval()

# bert_output=bert_model(return_dict['input_ids'],attention_mask=return_dict['attention_mask'])
# print(bert_output[0].shape)

# print(return_dict['input_ids'].shape)
# print(return_dict['attention_mask'].shape)
# print(return_dict['token_type_ids'].shape)
# print(return_dict['video_feat'].shape)
# print(return_dict['video_attn_mask'].shape)
# print(return_dict['label'].shape)









# #basic file imports
# csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
# csv_data=pd.read_csv(csv_file)
# base_folder="/data/digbose92/ads_complete_repo/ads_features/clip_embeddings/jwt_ads_of_world"
# embedding_file="/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs_0.5.pkl"
# label_map={'No transition':0,'Transition':1}
# num_classes=2
# audio_max_length=14
# video_max_length=333
# task_name='Transition_val'

# #dataset
# dataset=SAIM_single_task_dataset_audio_visual(csv_data,
#                                             base_folder,
#                                             embedding_file,
#                                             label_map,num_classes,
#                                             audio_max_length,
#                                             video_max_length,
#                                             task_name)
# #dataloader
# dataloader=DataLoader(dataset,batch_size=2,shuffle=True)

# #audio, video features with label and attention mask over audio and video
# aud_feat,vid_feat,label,attn_mask_audio,attn_mask_video=next(iter(dataloader))


# #audio, video features with label and attention mask over audio and video
# print(aud_feat.shape)
# print(vid_feat.shape)
# print(label.shape)
# print(attn_mask_audio.shape)
# print(attn_mask_video.shape)














    


    





         





    







    



