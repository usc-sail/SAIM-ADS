import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle
import numpy as np
#append path of datasets and models 
sys.path.append(os.path.join('..', '..','datasets'))
sys.path.append(os.path.join('..', '..','models'))
sys.path.append(os.path.join('..', '..','configs'))
sys.path.append(os.path.join('..', '..','losses'))
sys.path.append(os.path.join('..', '..','optimizers'))
sys.path.append(os.path.join('..', '..','utils'))
sys.path.append(os.path.join('..'))

#import all libraries 
import random
from ast import literal_eval
import torch
import yaml
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataset import *
from loss_functions import *
from perceiver_model import *
from optimizer import *
from metrics import calculate_stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
from evaluate_model import *
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
from transformers import BertTokenizer, BertModel, BertConfig
import json
import pickle 


def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

def test_model_topic(model_filename,config_data,device,seed_value):

    #fix seed for reproducibility
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #model filename and transfer to device
    model=torch.load(model_filename)
    model.to(device)

    csv_file=config_data['data']['csv_file']
    csv_data=pd.read_csv(csv_file)
    base_folder=config_data['data']['base_folder']
    transcript_file=config_data['data']['transcript_file']

    #test_data,train_data,val_data
    test_data=csv_data[csv_data['Split']=='test']
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']
    max_text_length=config_data['parameters']['text_max_length']
    max_video_length=config_data['parameters']['video_max_length']
    batch_size=config_data['parameters']['batch_size']
    num_epochs=config_data['parameters']['epochs']
    num_workers=config_data['parameters']['num_workers']
    model_name=config_data['model']['model_name']
    n_classes=config_data['model']['n_classes']
    task_name=config_data['parameters']['task_name']
    tokenizer=BertTokenizer.from_pretrained(model_name)

    #label map
    topic_file=config_data['data']['topic_file']
    with open(topic_file,'r') as f:
        label_map=json.load(f)

    #model instantiate
    model=torch.load(model_filename)
    model.to(device)

    test_ds=SAIM_single_task_dataset_visual_text_shot_level(test_data,
                                                transcript_file,
                                                tokenizer,
                                                base_folder,
                                                label_map,
                                                n_classes,
                                                max_text_length,
                                                max_video_length,
                                                task_name
                                                )

    test_dl=DataLoader(test_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    
    #loss function
    criterion=multi_class_cross_entropy_loss(device)

    test_loss,test_acc,test_f1,logits_array,clip_key_list=gen_validate_score_text_visual_perceiver_single_task_topic(model,test_dl,device,criterion)
    print(logits_array.shape)
    print(len(clip_key_list))

    print('Test loss: ',test_loss)
    print('Test accuracy: ',test_acc)
    print('Test f1 score: ',test_f1)

    return(test_loss,test_acc,test_f1,logits_array,clip_key_list)

def test_soc_msg_tone_transition_model(model_filename,config_data,device,seed_value):

    #fix seed for reproducibility
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #model filename and transfer to device
    model=torch.load(model_filename)
    model.to(device)

    #test_data,train_data,val_data
    csv_file=config_data['data']['csv_file']
    csv_data=pd.read_csv(csv_file)
    base_folder=config_data['data']['base_folder']
    transcript_file=config_data['data']['transcript_file']
    test_data=csv_data[csv_data['Split']=='test']
    max_text_length=config_data['parameters']['text_max_length']
    max_video_length=config_data['parameters']['video_max_length']
    batch_size=config_data['parameters']['batch_size']
    num_epochs=config_data['parameters']['epochs']
    num_workers=config_data['parameters']['num_workers']
    model_name=config_data['model']['model_name']
    n_classes=config_data['model']['n_classes']
    task_name=config_data['parameters']['task_name']
    tokenizer=BertTokenizer.from_pretrained(model_name)

    #label map
    if(task_name=='social_message'):
        label_map={'No':0,'Yes':1}
        social_list=test_data['social_message'].tolist()
        social_list=[label_map[i] for i in social_list]
        counter_social_message=dict(Counter(social_list))
        majority_class=max(counter_social_message,key=counter_social_message.get)
        majority_class_accuracy=counter_social_message[majority_class]/len(social_list)
        majority_class_labels=[majority_class]*len(social_list)
        f1_majority_class=f1_score(social_list,majority_class_labels,average='macro')

        print('Majority class accuracy: ',majority_class_accuracy)
        print('F1 score with majority class labels: ',f1_majority_class)

    else:
        label_map={'No transition':0,'Transition':1}
        transition_list=test_data['Transition_val'].tolist()
        transition_list=[label_map[i] for i in transition_list]
        counter_transition=dict(Counter(transition_list))
        majority_class=max(counter_transition,key=counter_transition.get)
        majority_class_accuracy=counter_transition[majority_class]/len(transition_list)
        majority_class_labels=[majority_class]*len(transition_list)
        f1_majority_class=f1_score(transition_list,majority_class_labels,average='macro')

        print('Majority class accuracy: ',majority_class_accuracy)
        print('F1 score with majority class labels: ',f1_majority_class)

    test_ds=SAIM_single_task_dataset_visual_text_shot_level(test_data,
                                                transcript_file,
                                                tokenizer,
                                                base_folder,
                                                label_map,
                                                n_classes,
                                                max_text_length,
                                                max_video_length,
                                                task_name
                                                )
    test_dl=DataLoader(test_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['test_shuffle'],
                        num_workers=num_workers)
    
    criterion = binary_cross_entropy_loss(device,pos_weights=None)

    test_loss,test_acc,test_f1,logits_array,clip_key_list=gen_validate_score_text_visual_perceiver_single_task_soc_message_tone(model,test_dl,device,criterion)

    print('Test loss: ',test_loss)
    print('Test accuracy: ',test_acc)
    print('Test f1 score: ',test_f1)

    return(test_loss,test_acc,test_f1,logits_array,clip_key_list)
    
   
argparser = argparse.ArgumentParser()
argparser.add_argument('--log_dir', required=True)
argparser.add_argument('--model_dir', required=True)
argparser.add_argument('--folder_name', required=True)
argparser.add_argument('--json_file', required=True)
argparser.add_argument('--save_preds', required=True)
argparser.add_argument('--save_preds_dir', required=True)

args=argparser.parse_args()

#avalon location
csv_file_loc="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
base_folder="/proj/digbose92/ads_repo/embeddings/shot_features/clip_features_4fps"
transcript_file="/proj/digbose92/ads_repo/transcripts/en_combined_transcripts.json"
topic_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"

#log dir, model dir, folder name, json file
log_dir=args.log_dir
model_dir=args.model_dir
folder_name=args.folder_name
json_file=args.json_file
save_preds=args.save_preds
save_preds_dir=args.save_preds_dir

log_subfolder=os.path.join(log_dir,folder_name)
model_subfolder=os.path.join(model_dir,folder_name)
log_file_list=os.listdir(log_subfolder)
model_file_list=os.listdir(model_subfolder)

if(args.save_preds=='True'):
    #save_preds=True
    dest_filename=os.path.join(save_preds_dir,os.path.splitext(json_file.split("/")[-1])[0]+".pkl")

with open(json_file,'r') as f:
    run_data=json.load(f)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_list_f1=[]
_list_acc=[]
dict_tot={}

for run in list(run_data.keys()):

    run_dict=run_data[run]

    seed=run_dict['seed']

    timestamp=run_dict['timestamp']

    log_timestamp_folder=[filename for filename in log_file_list if timestamp in filename][0]
    yaml_file=os.path.join(log_subfolder,log_timestamp_folder,log_timestamp_folder+".yaml")

    model_timestamp_folder=[filename for filename in model_file_list if timestamp in filename][0]
    model_filename=os.path.join(model_subfolder,model_timestamp_folder,model_timestamp_folder+"_best_model.pt")

    config_data=load_config(yaml_file)

    #specific to text_shot_level_visual_perceiver_single_task
    config_data['data']['csv_file']=csv_file_loc
    config_data['data']['base_folder']=base_folder
    config_data['data']['transcript_file']=transcript_file

    if('topic_file' in config_data['data']):
        config_data['data']['topic_file']=topic_file
    
    #print(timestamp,seed,,model_filename)
    if(os.path.exists(yaml_file) and os.path.exists(model_filename)):
        #print(yaml_file,model_filename)
        if(config_data['parameters']['task_name']=='Topic'):
            test_loss,test_acc,test_f1,logits_array,clip_key_list=test_model_topic(model_filename,config_data,device,seed)
        elif((config_data['parameters']['task_name']=='social_message') or (config_data['parameters']['task_name']=='Transition_val')):
            test_loss,test_acc,test_f1,logits_array,clip_key_list=test_soc_msg_tone_transition_model(model_filename,config_data,device,seed)

        print('Run: %s, Seed: %d, Test loss: %.4f, Test accuracy: %.4f, Test f1 score: %.4f'%(run,seed,test_loss,test_acc,test_f1))

        _list_f1.append(test_f1)
        _list_acc.append(test_acc)

    dict_temp={'seed':seed,
               'test_loss':test_loss,
               'test_acc':test_acc,
               'test_f1':test_f1,
               'logits':logits_array,
                'clip_key':clip_key_list,
                'timestamp':timestamp
               }
    dict_tot[run]=dict_temp

print('Mean test f1 score: ',100*np.mean(_list_f1))
print('Mean test accuracy: ',100*np.mean(_list_acc))

#standard deviation
print('Standard deviation of test f1 score: ',100*np.std(_list_f1))
print('Standard deviation of test accuracy: ',100*np.std(_list_acc))

dict_tot['mean_test_f1']=np.mean(_list_f1)
dict_tot['mean_test_acc']=np.mean(_list_acc)
dict_tot['std_test_f1']=np.std(_list_f1)
dict_tot['std_test_acc']=np.std(_list_acc)

# print(dest_filename)
# if(save_preds==True):
# with open(dest_filename,'wb') as f:
#     pickle.dump(dict_tot,f)
    
