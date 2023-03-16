import os 
import numpy as np 
import pandas as pd 
import json 
import pickle
import multiprocessing as mp
import argparse 

#folder path and option 
#base folder + scene folder 
base_scene_folder="/data/digbose92/ads_complete_repo/ads_videos/shot_folder"
scene_detect_option="PySceneDetect"
destination_scenes_folder=os.path.join(base_scene_folder,scene_detect_option)
csv_scenes_base_folder="/data/digbose92/ads_complete_repo/ads_videos/shot_csv_folder"
csv_scenes_folder=os.path.join(csv_scenes_base_folder,scene_detect_option)

#read the file list 
file_list="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/lomond_available_file_path.txt"
with open(file_list,"r") as f:
    file_list=f.readlines()

file_list=[f.strip() for f in file_list]
file_list=[f.split("\n") for f in file_list if f.endswith(".mp4")]

def extract_scene_clips(idx):

    vid_file=file_list[idx]
    file_key=os.path.splitext(vid_file.split("/")[-1])[0]
    subfolder=os.path.join(destination_scenes_folder,file_key)
    csv_scenes_file=os.path.join(csv_scenes_folder,file_key+".csv")

    if(os.path.exists(subfolder) is False):
        os.mkdir(subfolder)
        scene_detect_command="scenedetect --input "+vid_file+ " -s "+csv_scenes_file+" detect-content list-scenes split-video -o "+subfolder
        os.system(scene_detect_command)

def main(args):

    pool = mp.Pool(args.nj)
    pool.map(extract_scene_clips, list(range(len(file_list))))
    pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nj', default=16, type=int, help='number of parallel processes')
    args = parser.parse_args()
    main(args)
# print(len(file_list))

#print(video_file)


#print(len(set(video_file_list) & set(file_keys)))
#print(len(set(video_file_list)))
#print(len(set(file_keys)))




