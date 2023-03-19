import os 
import pandas as pd 
import argparse
import timm
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2 
import math
import pickle  

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def run_frame_wise_feature_inference(model,processor,filename,device,dim=768,desired_frameRate=4):
  
    vcap=cv2.VideoCapture(filename)
    frameRate = vcap.get(5)
    intfactor=math.ceil(frameRate/desired_frameRate)
    feature_list=np.zeros((0,dim))
    frame_id=0

    length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    tensor_list=[]

    while True:
        ret, frame = vcap.read()
        if(ret==True):
            if (frame_id % intfactor == 0):
                #print(frame_id)
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=Image.fromarray(frame)
                inputs = processor(frame,return_tensors='pt')
                #print(inputs.keys())
                inputs['pixel_values']=inputs['pixel_values'].to(device)
                
                with torch.no_grad():
                    outputs=model(**inputs,output_hidden_states=True)
                
                hidden_states=outputs['hidden_states']
                

                cls_embedding=hidden_states[-1][:,0,:].cpu().numpy()
                feature_list=np.vstack([feature_list,cls_embedding]) #add the features to the numpy array
                
                torch.cuda.empty_cache()
            frame_id=frame_id+1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return feature_list, frame_id


#argparse arguments 
parser = argparse.ArgumentParser(description='Extract vit base features from a video file')
parser.add_argument('--feature_folder', type=str, help='path to the destination feature folder')
parser.add_argument('--video_folder', type=str, help='path to the destination feature folder')
parser.add_argument('--model_name', type=str, help='path to the model name')
parser.add_argument('--video_type',type=str,default='shot',help='path to the video type')
parser.add_argument('--shot_subfolder', type=str, help='path to the shot subfolder')
args=parser.parse_args()

model_name=args.model_name
shot_subfolder=args.shot_subfolder
video_type=args.video_type
video_folder=args.video_folder

#declare vit models from timm specification
print('Loading model')
model = ViTForImageClassification.from_pretrained(model_name)
model.config.return_dict=True
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
model.eval()

processor=ViTFeatureExtractor.from_pretrained(model_name)
#print layer wise names
#declare the transforms


print('Loaded model')
#load the model along with the logits
# h1 = model.pre_logits.register_forward_hook(getActivation('pre_logits'))

if(video_type=='shot'):
    #shot subfolder
    #shot_subfolder=os.path.join(args.feature_folder,args.shot_subfolder)
    shot_folder_name=os.path.join(video_folder,shot_subfolder)
    #video file list
    video_file_list=os.listdir(shot_folder_name)

    for video_file in tqdm(video_file_list):

        video_subfolder=os.path.join(shot_folder_name,video_file)
        destination_file=os.path.join(args.feature_folder,video_file+'.pkl')

        shot_list=os.listdir(video_subfolder) #list of shots
        shot_dict=dict()

        for shot_file in tqdm(shot_list):
            
            shot_filename=os.path.join(video_subfolder,shot_file)
            feat_list,_=run_frame_wise_feature_inference(model,processor,shot_filename,device)

            shot_dict[shot_file]=feat_list

        #save the shot_dict

        with open(destination_file,'wb') as f:
            pickle.dump(shot_dict,f)

            
