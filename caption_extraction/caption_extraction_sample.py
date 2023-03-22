import os 
from lavis.models import model_zoo
import cv2 
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm 
import pandas as pd
#print(model_zoo)

#consider a sample video folder with shots 
caption_folder="/bigdata/digbose92/ads_data/ads_complete_repo/ads_captions"
#for each shot first check how the captioning outputs change over the entire duration
shot_folder="/bigdata/digbose92/ads_data/ads_complete_repo/ads_videos/shot_folder/PySceneDetect"
destination_folder="/bigdata/digbose92/ads_data/ads_complete_repo/ads_captions"
subfolder_name="3LB9sKseRGg"
subfolder_path=shot_folder+"/"+subfolder_name
file_list=os.listdir(subfolder_path)
dest_subfolder_path=destination_folder+"/"+subfolder_name
if(os.path.exists(dest_subfolder_path)==False):
    os.mkdir(dest_subfolder_path)
#for each shot sample a frame from the center of the shot randomly 

#load the modeldevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

#run the blip image captioning model at the center of the shot 
for video_file in tqdm(file_list):

    cap = cv2.VideoCapture(os.path.join(subfolder_path,video_file))
    print('Processing video: ', video_file)

    frame_number=0
    caption_list=[]
    frame_number_list=[]

    while(cap.isOpened()):

        ret, frame = cap.read()
        #print
        if ret == False:
            break

        #convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_image=Image.fromarray(frame)

        #IMAGE AND CAPTION GENERATION
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption=model.generate({"image": image})

        #frame number and caption list
        caption_list.append(caption)
        frame_number_list.append(frame_number)

        frame_number+=1
        print('frame_number: ', frame_number, 'caption: ', caption)

    cap.release()

    #save the captions and frame numbers in a file
    file_name=os.path.splitext(video_file.split("/")[-1])[0]+".csv"
    file_path=dest_subfolder_path+"/"+file_name

    df=pd.DataFrame({'frame_number':frame_number_list, 'caption':caption_list})

    df.to_csv(file_path, index=False)









#save the image snapshots and captions in a folder