import os 
from lavis.models import model_zoo
import cv2 
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm 
import pandas as pd
import argparse 

#also save the frames that are being used to extract the captions
parser = argparse.ArgumentParser(description='Extract captions from a shots in a specific video')
parser.add_argument('--caption_folder',type=str,help='folder where the captions are stored')
parser.add_argument('--image_folder',type=str,help='folder where the images are stored')
parser.add_argument('--model_name',type=str,help='blip_caption')
parser.add_argument('--model_type',type=str,help='large_coco')
parser.add_argument('--shot_folder',type=str,help='folder where the shots are stored')
parser.add_argument('--shard_file',type=str,help='shard file containing the list of videos')

args = parser.parse_args()
caption_folder = args.caption_folder
image_folder = args.image_folder
model_name = args.model_name
model_type = args.model_type
shot_folder = args.shot_folder
shard_file = args.shard_file

#load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)

with open(shard_file) as f:
    subfolder_list = f.readlines()

subfolder_list = [x.strip().split("\n")[0] for x in subfolder_list]

#subfolder_list=os.listdir(shot_folder)

for subfolder in tqdm(subfolder_list):

    dest_file=caption_folder+"/"+subfolder+".csv"

    if(os.path.exists(dest_file) is False):
        #shot subfolder
        subfolder_path=shot_folder+"/"+subfolder

        #image subfolder
        image_subfolder_path=image_folder+"/"+subfolder
        if(os.path.exists(image_subfolder_path)==False):
            os.mkdir(image_subfolder_path)

        #list of all shots 
        shot_file_list=os.listdir(subfolder_path)
        frame_shot_list=[]
        caption_list=[]
        shot_key_list=[]

        for video_file in tqdm(shot_file_list):
            
            #video capture
            cap = cv2.VideoCapture(os.path.join(subfolder_path,video_file))

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the middle frame
            middle_frame = int(total_frames / 2)

            # Set the video's current frame to the middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

            success, frame = cap.read() #middle frame is read 


            if(success):
                #save the frame current being used for the caption extraction
                frame_file=image_subfolder_path+"/"+'frame_num_'+str(middle_frame)+'.jpg'

                #save the frame
                cv2.imwrite(frame_file,frame)

                #extract caption for the frame 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_image=Image.fromarray(frame)

                #IMAGE AND CAPTION GENERATION
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                caption=model.generate({"image": image})

            else:
                caption=""
                break 

            #append the frame and the caption to the list
            frame_shot_list.append(middle_frame)
            caption_list.append(caption[0])
            shot_key_list.append(video_file)


        #dataframe for the current shot file 

        df=pd.DataFrame({"Frame_number":frame_shot_list,"Caption":caption_list,"Shot_key":shot_key_list})
        df.to_csv(caption_folder+"/"+subfolder+".csv",index=False)

















