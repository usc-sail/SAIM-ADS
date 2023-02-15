import pandas as pd 
import numpy as np 
import pickle 
import os 
from random import sample

def generate_train_test_val_split(indices,
                    train_split=0.7):

    #list of train indices
    train_indices=sample(indices,int(train_split*len(indices)))
    #list of test and validation indices
    test_val_indices=list(set(indices)-set(train_indices))
    #list of validation indices
    val_indices=sample(test_val_indices,int(0.33*len(test_val_indices))) #10% is validation, 20% is test
    #list of test indices
    test_indices=list(set(test_val_indices)-set(val_indices))

    print(len(train_indices),len(val_indices),len(test_indices))

    #intersection between train and val indices
    print(len(list(set(train_indices) & set(val_indices))))
    print(len(list(set(train_indices) & set(test_indices))))
    print(len(list(set(val_indices) & set (test_indices))))

    return(train_indices,val_indices,test_indices)

#csv file for tone transition and social message
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone.csv"
filename=csv_file.split("/")[-1]
csv_data=pd.read_csv(csv_file)
train_indices,val_indices,test_indices=generate_train_test_val_split(range(csv_data.shape[0]))

split_list=['NA']*(csv_data.shape[0])
split_list=np.array(split_list)

split_list[train_indices]='Train'
split_list[val_indices]='Val'
split_list[test_indices]='Test'

csv_data['Split']=list(split_list)

#drop the column marked Unnamed: 0
csv_data=csv_data.drop(columns=['Unnamed: 0'])
print(csv_data.head(5))
csv_data.to_csv('../data/'+filename.split(".")[0]+"_train_test_val.csv",index=False)




