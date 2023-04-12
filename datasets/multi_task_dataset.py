from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os 
import torch 
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import pickle
import json 

######### multi task dataset declaration #########

class Multi_Task_Shot_Dataset(Dataset):

    def __init__(self,csv_data,max_length,task_label_map,base_folder):

        self.csv_data=csv_data
        self.max_length=max_length
        self.clip_feature_list=self.csv_data['clip_feature_path'].tolist()
        self.task_label_map=task_label_map
        self.base_folder=base_folder
        #task_label_map has a nested dictionary structure with task names as keys followed by label mapping

        #shot feature list
        self.shot_feature_list=[os.path.join(self.base_folder,file.split("/")[-1]) for file in self.clip_feature_list]

    #length of the csv data
    def __len__(self):
        return(len(self.csv_data))
    
    # length of the padded feature vector
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

        #get the label for the tasks 
        label_dict={}
        for k in self.task_label_map.keys():
            label_dict[k]=self.task_label_map[k][self.csv_data[k].iloc[idx]]


        return(shot_feat_padded,attention_mask,label_dict)
            

csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
csv_data=pd.read_csv(csv_file)
base_folder="/data/digbose92/ads_complete_repo/ads_features/shot_embeddings/clip_features_4fps/"
train_data=csv_data[csv_data['Split']=='train']
max_length=20
task_label_map={
    'social_message':{'No':0,'Yes':1},
    'Topic': {"Games": 0,"Household": 1,"Services": 2,"Misc": 3,
    "Sports": 4,
    "Banking": 5,
    "Clothing": 6,
    "Industrial and agriculture": 7,
    "Leisure": 8,
    "Publications media": 9,
    "Health": 10,
    "Car": 11,
    "Electronics": 12,
    "Cosmetics": 13,
    "Food and drink": 14,
    "Awareness": 15,
    "Travel and transport": 16,
    "Retail": 17
}
}

train_dataset=Multi_Task_Shot_Dataset(train_data,max_length,task_label_map,base_folder)

#load the data
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)

shot_feat,attn_mask,label_dict=next(iter(train_loader))

print(shot_feat.shape)
print(attn_mask.shape)
print(label_dict)
    

        

