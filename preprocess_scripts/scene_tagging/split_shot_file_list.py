import os 

dest_folder="/data/digbose92/ads_complete_repo/ads_videos/shot_folder/shot_folder_splits"
txt_file_list="/data/digbose92/ads_complete_repo/ads_videos/shot_folder/shot_list.txt"
with open(txt_file_list,'r') as f:
    shot_list=f.readlines()

shot_list=[shot.strip().split("\n")[0] for shot in shot_list]

#split into chunks of 2000
chunk_size=2000
chunk_list=[shot_list[i:i+chunk_size] for i in range(0,len(shot_list),chunk_size)]

#write to file
for i,chunk in enumerate(chunk_list):
    
    dest_filename=os.path.join(dest_folder,"shot_list_"+str(i)+".txt")
    with open(dest_filename,'w') as f:
        for shot in chunk:
            f.write(shot+"\n")

