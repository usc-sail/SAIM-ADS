from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os 
import torch 
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import pickle

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



# #test the dataset
# csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_data.csv"
# csv_data=pd.read_csv(csv_file)
# label_map={'No transition':0,'Transition':1}
# # csv_data=pd.read_csv(csv_file)
# # label_map={'No transition':0,'Transition':1}
# num_class=2
# max_length=333
# fps=4
# base_fps=24
# task_name="Transition_val"

# #single task dataset and dataloader
# saim_single_task_dataset=SAIM_single_task_dataset(csv_data=csv_data,
#                                                 label_map=label_map,
#                                                 num_classes=num_class,
#                                                 max_length=max_length,
#                                                 fps=fps,
#                                                 base_fps=base_fps,
#                                                 task_name=task_name)

# saim_single_task_dataloader=DataLoader(saim_single_task_dataset,batch_size=8,shuffle=True)
# clip_feat,ret_label,attention_mask = next(iter(saim_single_task_dataloader))

# print(clip_feat.shape)
# print(ret_label.shape)
# print(attention_mask)

# #count number of ones in the attention mask per row
# count_ones = torch.sum(attention_mask, dim=1)
# print(count_ones)



# saim_ads_tone_clip_ds=SAIM_ads_tone_clip_features_dataset(csv_data=csv_data,label_map=label_map,
#                                 num_classes=num_class,
#                                 max_length=max_length,
#                                 fps=fps,
#                                 base_fps=base_fps)

# saim_ads_tone_clip_dl=DataLoader(saim_ads_tone_clip_ds,batch_size=8,shuffle=True)

# clip_feat,trans_array,feat_len=next(iter(saim_ads_tone_clip_dl))

# print(clip_feat.shape)
# print(trans_array.shape)
# print(feat_len)
# print(trans_array)



    







    



