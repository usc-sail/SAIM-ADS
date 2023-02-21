#helper script to extract frame wise features using CLIP visual encoder
import os 
import clip 
import torch  
import pickle
from PIL import Image
import pandas as pd 
import glob
from tqdm import tqdm
import cv2 
from tqdm import tqdm
import numpy as np  
#extract from the videos directly 

def generate_video_prediction(model,preprocess,device,video_file,batch_size=32):

    vcap=cv2.VideoCapture(video_file)
    feature_list=np.zeros((0,512))
    frame_id=1
    length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list=[]
    block_count=0
    #print(video_file)
    while True:
        ret, frame = vcap.read()
        if(ret==True):
            
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #convert BGR to RGB
            frame=Image.fromarray(frame) #convert BGR image to PIL Image
            frame = preprocess(frame).unsqueeze(0).to(device) #preprocess the frame


            #concatenate till batch size is divisible by 32 and then do a forward pass 
            if((frame_id%batch_size==0) or (frame_id==length)):
                frame_list.append(frame)
                frame_comb=torch.cat(frame_list,dim=0)
                block_count=block_count+1
                #extract the image features
                #print(frame_comb.size())
                with torch.no_grad():
                    image_features = model.encode_image(frame_comb)

                frame_list=[]
                image_features=image_features.cpu().numpy()
                feature_list=np.vstack([feature_list,image_features])
                #print(frame_id,feature_list.shape)
                #add this chunk to an empty array 
            else:
                frame_list.append(frame)
                
            #print(frame_id,len(frame_list))
            #image_features=image_features.cpu().numpy().squeeze(0)
            #feature_list.append(image_features)
            
            frame_id=frame_id+1
        else:
            #print(frame_id+1)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #print(block_count,feature_list.shape,length,frame_id)
    #image_features=np.array(feature_list)
    return(feature_list)



def generate_predictions_clip_ads(model,preprocess,frames_location,key,device,batch_size=32):

    #using dumped frames

    subfolder=os.path.join(frames_location,key)
    file_list=os.listdir(subfolder)
    flist=[int(f.split(".")[0][3:]) for f in file_list]
    flist=sorted(flist)  
    similarity_list=[]
    dict_frame_list=dict()
    length=(len(flist))
    frame_id=1
    frame_list=[]
    key_list=[]
    feature_list=np.zeros((0,512))

    for image_file in flist:
        #print(image_file)

        if(image_file<10):
            img_file=os.path.join(subfolder,'out0'+str(image_file)+".jpg")
        else:
            img_file=os.path.join(subfolder,'out'+str(image_file)+".jpg")
        
        image=cv2.imread(img_file)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(image) 
        image = preprocess(image).unsqueeze(0).to(device)
        frame_list.append(image)

        key_list.append(image_file)

        if((frame_id%batch_size==0) or (frame_id==length)):
                
                frame_comb=torch.cat(frame_list,dim=0)
                with torch.no_grad():
                    image_features = model.encode_image(frame_comb)

                image_features=image_features.cpu().numpy()
                feature_list=np.vstack([feature_list,image_features])

                #for i,key in enumerate(key_list):
                #    dict_frame_list[key]=image_features[i,:]

                frame_list=[]
                #key_list=[]

        frame_id=frame_id+1
                #print(frame_id,feature_list.shape)
                #add this chunk to an empty array 
            #else:
            #    frame_list.append(frame)

        #how would you run inference on the features with group of keys and then save those keys to dictionary ?


        #encode the features with keys in the dictionary
        

         
        # with torch.no_grad():
        #image_features = model.encode_image(image)
    dict_frame_list={'Features':feature_list,'Keys':key_list}
    assert(dict_frame_list['Features'].shape[0]==len(key_list))
    #print(key_list[0:1000])

    return(dict_frame_list)
        

def run_ads_video_inference(video_list,model,device,preprocess,destination_folder):


    for vid in tqdm(video_list):
        vid_key=vid.split("/")[-1]
        vid_destination_file=os.path.join(destination_folder,vid_key.replace(".mp4",".pkl")) #pickle file to store the features 
        dict_temp=dict()
        if(os.path.exists(vid_destination_file) is False):
            
            clip_features=generate_video_prediction(model,preprocess,device,vid)
            dict_temp['Features']=clip_features
            #print(clip_features.shape)
            with open(vid_destination_file,"wb") as f:
                pickle.dump(dict_temp,f)
            #np.save(vid_destination_file,clip_features)
        #else:
        #    print('File:%s exists' %(vid_destination_file))

def run_frames_inference(frames_folder,model,device,preprocess,destination_folder):

    for subfolder in tqdm(os.listdir(frames_folder)):
        vid_destination_file=os.path.join(destination_folder,subfolder+".pkl")
        if(os.path.exists(vid_destination_file) is False):
            dict_features=generate_predictions_clip_ads(model,preprocess,frames_folder,subfolder,device)
            #print(len(dict_features))
            with open(vid_destination_file,"wb") as f:
                pickle.dump(dict_features,f)


def run_video_inference_updated(video_list,model,device,preprocess,destination_folder):

    cnt_num_files=0
    for vid in tqdm(video_list):
        vid_key=vid.split("/")[-1]
        if(vid_key.endswith(".mp4") is True):
            vid_destination_file=os.path.join(destination_folder,vid_key.replace(".mp4",".pkl")) #pickle file to store the features 
        elif(vid_key.endswith(".webm") is True):
            vid_destination_file=os.path.join(destination_folder,vid_key.replace(".webm",".pkl"))
        else:
            vid_destination_file=os.path.join(destination_folder,vid_key.replace(".mkv",".pkl"))

        dict_temp=dict()

        #if(os.path.exists(vid_destination_file) is False):
                
        clip_features=generate_video_prediction(model,preprocess,device,vid)
        print(vid_destination_file,clip_features.shape)
        dict_temp['Features']=clip_features
        with open(vid_destination_file,"wb") as f:
            pickle.dump(dict_temp,f)

    #print('Number of files processed for the updated set: %d' %(cnt_num_files))

if __name__=='__main__':

    feature_destination_folder="/data/digbose92/ads_complete_repo/ads_features/clip_embeddings/jwt_ads_of_world"
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    feature_file="/data/digbose92/ads_complete_repo/ads_codes/updated_pkl_files/corrected_files/zero_clip_features_files.pkl"
    #"/data/digbose92/ads_complete_repo/ads_codes/updated_pkl_files/corrected_files/file_list_clip_features_extraction_remaining.pkl"
    with open(feature_file, "rb") as f:
        video_file_list=pickle.load(f)
    
    run_video_inference_updated(video_file_list,model,device,preprocess,feature_destination_folder)




        






