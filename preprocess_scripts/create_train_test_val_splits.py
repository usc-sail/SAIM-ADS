import pandas as pd 
import numpy as np 
import pickle 
import os 
from random import sample
from collections import Counter 
import random 

#fix seed here
seed_value=42
np.random.seed(seed_value) # cpu vars
random.seed(seed_value) # Python


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

def process_csv(csv_data):

    #remove the trailing spaces in the column entries and newline entries 

    column_list=list(csv_data.columns)
    # csv_data[column_list] = csv_data[column_list].apply(lambda x: x.str.strip())
    # csv_data[column_list] = csv_data[column_list].apply(lambda x: x.str.split('\n')[0])
    #print(csv_data.head(5))
    csv_data['video_file']=csv_data['video_file'].apply(lambda x:x.split('\n')[0])
    return(csv_data)


#csv file for tone transition and social message
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone.csv"
filename=csv_file.split("/")[-1]
csv_data=pd.read_csv(csv_file)
train_indices,val_indices,test_indices=generate_train_test_val_split(range(csv_data.shape[0]))

split_list=[None]*(csv_data.shape[0])
split_list=np.array(split_list)

#print(train_indices)
split_list[train_indices]='train'
split_list[val_indices]='val'
split_list[test_indices]='test'

csv_data['Split']=list(split_list)


#drop the column marked Unnamed: 0
csv_data=csv_data.drop(columns=['Unnamed: 0'])
csv_data=process_csv(csv_data)

#distribution of train, test and val for transition and no-transition

csv_data_train=csv_data[csv_data['Split']=='train']
csv_data_val=csv_data[csv_data['Split']=='val']
csv_data_test=csv_data[csv_data['Split']=='test']


#train, val, test statistics
transition_csv_data_train=list(csv_data_train['Transition_val'])
transition_csv_data_val=list(csv_data_val['Transition_val'])
transition_csv_data_test=list(csv_data_test['Transition_val'])

social_message_train=list(csv_data_train['social_message'])
social_message_val=list(csv_data_val['social_message'])
social_message_test=list(csv_data_test['social_message'])

print(Counter(transition_csv_data_train))
print(Counter(transition_csv_data_val))
print(Counter(transition_csv_data_test))

print(Counter(social_message_train))
print(Counter(social_message_val))
print(Counter(social_message_test))

#transition results

# Counter({'No transition': 3205, 'Transition': 2715}): Train
# Counter({'No transition': 450, 'Transition': 387}): Val
# Counter({'No transition': 927, 'Transition': 774}): Test

#social message results

#Counter({'No': 5381, 'Yes': 539}) : Train
#Counter({'No': 753, 'Yes': 84}): Val
#Counter({'No': 1563, 'Yes': 138}): Test


print(csv_data.head(5))
csv_data.to_csv('../data/'+filename.split(".")[0]+"_train_test_val.csv",index=False)




