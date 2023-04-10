import os 
import pandas as pd 

folder="/project/shrikann_35/dbose/codes/ads_codes/caption_split_file"
caption_folder="/scratch1/dbose/ads_repo/captions/csv_files"
csv_file_list=os.listdir(caption_folder)
file_keys=[os.path.splitext(f)[0] for f in csv_file_list]
file_list=os.listdir(folder)

len_total_content=0
total_list_of_files=[]

for file in file_list:
    file_path=folder+"/"+file
    
    with open(file_path) as f:
        content = f.readlines()

    total_list_of_files=total_list_of_files+[c.strip().split("\n")[0] for c in content]
    
    len_total_content=len_total_content+len(content)

print('Number of file:%d' %(len_total_content))

#number of intersection between the two lists
print('Number of intersection:%d' %(len(set(file_keys).intersection(set(total_list_of_files)))))

#difference between the two lists
diff_list=list(set(total_list_of_files)-set(file_keys))

# save the difference list
with open('/project/shrikann_35/dbose/codes/ads_codes/caption_split_file/incomplete_files_caption_extraction.txt','w') as f:
    for item in diff_list:
        f.write(item + "\n")



