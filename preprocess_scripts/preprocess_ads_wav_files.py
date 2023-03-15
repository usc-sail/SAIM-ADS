import os 
import numpy as np 


folder="/scratch2/dbose/ads_complete_dir/ads_wav_files/jwt_ads_of_world_wav_files"
shard_folder="/scratch2/dbose/ads_complete_dir/ads_wav_shards"

file_list=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".wav")]

#split the list into chunks of 1000

file_list_chunks=np.array_split(file_list,10)
print(file_list_chunks[0])

for i in range(len(file_list_chunks)):

    print("Processing chunk ",i)

    with open(os.path.join(shard_folder,"shard_"+str(i)+".txt"),"w") as f:
        for file in file_list_chunks[i]:
            f.write(file+"\n")
#print(file_list[0:5])
#def create_shards(folder,shard_folder):

    