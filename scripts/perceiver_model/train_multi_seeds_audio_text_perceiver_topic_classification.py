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

######## global config file declaration ########
config_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/configs/perceiver_configs/config_audio_text_perceiver_topic_multiple_seeds.yaml"
with open(config_file,'r') as f:
    config_data=yaml.safe_load(f)

#csv and embedding + topic files 
csv_file=config_data['data']['csv_file']
embedding_file=config_data['data']['embedding_file']
topic_file=config_data['data']['topic_file']
csv_data=pd.read_csv(csv_file)
task_name=config_data['parameters']['task_name']
transcript_file=config_data['data']['transcript_file']

#load the topic file 
with open(topic_file,'r') as f:
    label_map=json.load(f)

#parameters regarding number of classes, maximum audio length, maximum video length
max_audio_length=config_data['parameters']['audio_max_length']
max_text_length=config_data['parameters']['text_max_length']
batch_size=config_data['parameters']['batch_size']
num_epochs=config_data['parameters']['epochs']
num_workers=config_data['parameters']['num_workers']

#parameters regarding the perceiver model
audio_dim=config_data['model']['audio_dim']
text_dim=config_data['model']['text_dim']
dim=config_data['model']['dim']
queries_dim=config_data['model']['queries_dim']
depth=config_data['model']['depth']
num_latents=config_data['model']['num_latents']
cross_heads=config_data['model']['cross_heads']
latent_heads=config_data['model']['latent_heads']
cross_dim_head=config_data['model']['cross_dim_head']
latent_dim_head=config_data['model']['latent_dim_head']
latent_dim=config_data['model']['latent_dim']
weight_tie_layers=config_data['model']['weight_tie_layers']
seq_dropout_prob=config_data['model']['seq_dropout_prob']
n_classes=config_data['model']['n_classes']
use_queries=config_data['model']['use_queries']
model_name=config_data['model']['model_name']
model_type=config_data['model']['model_type']
option=model_type+"_"+config_data['parameters']['task_name']
multi_run_folder=config_data['output']['multiple_run_folder']
tokenizer=BertTokenizer.from_pretrained(model_name)

#create the model type folder inside the multi run folder
destination_run_folder=os.path.join(multi_run_folder,model_type)
if not os.path.exists(destination_run_folder):
    os.mkdir(destination_run_folder)

#create random seeds for multiple runs 
seed_list = [random.randint(1, 100000) for _ in range(5)]

test_loss_multiple_seeds_list=[]
test_f1_multiple_seeds_list=[]
test_acc_multiple_seeds_list=[]
val_best_f1_multiple_seeds_list=[]
val_best_acc_multiple_seeds_list=[]

#device definition
if(config_data['device']['is_cuda']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#seed value set 
dict_multiple_seeds={}

for i,seed in enumerate(seed_list):

    # global fixing here 
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Run number: %d with random seed: %d' %(i+1,seed))

    #define the datasets 
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']
    test_data=csv_data[csv_data['Split']=='test']

    #dataset definitions
    train_ds=SAIM_single_task_dataset_audio_text(train_data,
                                                transcript_file,
                                                tokenizer,
                                                embedding_file,
                                                label_map,
                                                n_classes,
                                                max_text_length,
                                                max_audio_length,
                                                task_name
                                                )

    val_ds=SAIM_single_task_dataset_audio_text(val_data,
                                                transcript_file,
                                                tokenizer,
                                                embedding_file,
                                                label_map,
                                                n_classes,
                                                max_text_length,
                                                max_audio_length,
                                                task_name
                                                )
    
    test_ds=SAIM_single_task_dataset_audio_text(test_data,
                                                transcript_file,
                                                tokenizer,
                                                embedding_file,
                                                label_map,
                                                n_classes,
                                                max_text_length,
                                                max_audio_length,
                                                task_name
                                                )
    
    #define the dataloaders
    train_dl=DataLoader(train_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['train_shuffle'],
                        num_workers=num_workers)

    val_dl=DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['val_shuffle'],
                        num_workers=num_workers)
    
    test_dl=DataLoader(test_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['test_shuffle'],
                        num_workers=num_workers)
    
    #define the model
    params_dict={'text_dim':text_dim,
                 'audio_dim':audio_dim,
                 'dim':dim,
                 'bert_model_name':model_name,
                 'queries_dim':queries_dim,
                 'num_classes':n_classes,
                 'depth':depth,
                 'num_latents':num_latents,
                 'cross_heads':cross_heads,
                 'latent_heads':latent_heads,
                 'cross_dim_head':cross_dim_head,
                 'latent_dim_head':latent_dim_head,
                 'latent_dim':latent_dim,
                 'weight_tie_layers':weight_tie_layers,
                 'seq_dropout_prob':seq_dropout_prob,
                 'use_queries':use_queries,}

    model=Perceiver_AudioText_model(**params_dict)


    #model parameters 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters: %d' %(params))
    model=model.to(device)
    
    ############################# loss function + optimizers definition here ################################
    if(config_data['loss']['loss_option']=='bce_cross_entropy_loss'):
        criterion = binary_cross_entropy_loss(device,pos_weights=None)

    elif(config_data['loss']['loss_option']=='cross_entropy_loss'):
        criterion=multi_class_cross_entropy_loss(device)

    if(config_data['optimizer']['choice']=='Adam'):
        optim_example=optimizer_adam(model,float(config_data['optimizer']['lr']))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename=timestr+'_'+option+'_log.logs'
    yaml_filename=timestr+'_'+option+'.yaml'

    log_model_subfolder=os.path.join(config_data['output']['log_dir'],option)
    if(os.path.exists(log_model_subfolder) is False):
        os.mkdir(log_model_subfolder)
    #create log folder associated with current model
    sub_folder_log=os.path.join(log_model_subfolder,timestr+'_'+option)
    if(os.path.exists(sub_folder_log) is False):
        os.mkdir(sub_folder_log)

    #create model folder associated with current model
    model_loc_subfolder=os.path.join(config_data['output']['model_dir'],option)
    if(os.path.exists(model_loc_subfolder) is False):
        os.mkdir(model_loc_subfolder)
    
    sub_folder_model=os.path.join(model_loc_subfolder,timestr+'_'+option)
    if(os.path.exists(sub_folder_model) is False):
        os.mkdir(sub_folder_model)

    #save the current config in the log_dir 
    yaml_file_name=os.path.join(sub_folder_log,yaml_filename)
    #print(yaml_file_name)
    with open(yaml_file_name, "w") as f:
        yaml.dump(config_data, f)

    #logger declaration #
    logger = log(path=sub_folder_log, file=filename)
    logger.info('Starting training')
    logger.info(config_data)

    #earky stoppping criteria
    early_stop_counter=config_data['parameters']['early_stop']
    print('Early stop criteria:%d' %(early_stop_counter))
    early_stop_cnt=0
    train_loss_stats=[]
    val_loss_stats=[]
    log_softmax= nn.Softmax(dim=-1)
    best_f1_score=0
    model.train(True)

    for epoch in range(1, num_epochs+1): #main outer loop


        train_loss_list=[]
        train_logits=[]
        step=0
        t = time.time()
        target_labels=[]
        pred_labels=[]
        val_loss_list=[]

        for id,(return_dict) in enumerate(tqdm(train_dl)):
            
            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            input_ids=return_dict['input_ids'].to(device)
            attention_mask=return_dict['attention_mask'].to(device)
            token_type_ids=return_dict['token_type_ids'].to(device)

            #return dict contains video features and attention mask
            audio_feat=return_dict['audio_feat'].float()

            audio_feat=audio_feat.to(device)
            audio_attn_mask=return_dict['audio_attn_mask'].to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            optim_example.zero_grad()
            logits=model(input_ids=input_ids,
                         audio_inputs=audio_feat,
                         text_mask=attention_mask,
                         audio_mask=audio_attn_mask)

            #loss calculation here
            loss = criterion(logits, label)
            train_logits=log_softmax(logits)
            y_pred=torch.max(train_logits, 1)[1]

            # Back prop.
            loss.backward()
            optim_example.step()

            train_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())

            step=step+1
            
            if(step%150==0):
                logger_step_dict={'Running_Train_loss':mean(train_loss_list)}
                logger.info("Training loss:{:.3f}".format(loss.item()))
                #wandb.log(logger_step_dict)

        target_label_np=torch.cat(target_labels).detach().numpy()
        pred_label_np=torch.cat(pred_labels).detach().numpy()

        #compute training accuracy and F1 score
        train_acc=accuracy_score(target_label_np,pred_label_np)
        train_f1=f1_score(target_label_np,pred_label_np,average='macro')

        logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
        logger.info('Epoch:{:d},Overall Training loss:{:.3f},Overall training Acc:{:.3f}, Overall F1:{:.3f}'.format(epoch,mean(train_loss_list),train_acc,train_f1))

        logger.info('Evaluating the dataset')
        val_loss,val_acc,val_f1=gen_validate_score_audio_text_perceiver_single_task_topic(model,val_dl,device,criterion)
        logger.info('Epoch:{:d},Overall Validation loss:{:.3f},Overall validation Acc:{:.3f}, Overall F1:{:.3f}'.format(epoch,val_loss,val_acc,val_f1))

        model.train(True)

        if(val_f1>best_f1_score):
            best_f1_score=val_f1
            logger.info('Saving the best model')
            torch.save(model, os.path.join(sub_folder_model,timestr+'_'+option+'_best_model.pt'))
            early_stop_cnt=0
        else:
            early_stop_cnt=early_stop_cnt+1
            
        if(early_stop_cnt==early_stop_counter):
            print('Validation performance does not improve for %d iterations' %(early_stop_counter))
            break

    #test the model 
    print('Training complete. Resuming testing with current seed')
    model.eval()

    #test loss, accuracy and F1 score
    test_loss,test_acc,test_f1=gen_validate_score_audio_text_perceiver_single_task_topic(model,test_dl,device,criterion)

    #current seed - test loss, accuracy and F1 score
    print('Current seed: %d, Test loss: %f, Test accuracy: %f, Test f1: %f' %(seed,test_loss,test_acc,test_f1))

    test_f1_multiple_seeds_list.append(test_f1)
    test_loss_multiple_seeds_list.append(test_loss)
    test_acc_multiple_seeds_list.append(test_acc)

    dict_temp={'seed':seed,
                'test_loss':test_loss,
                'test_acc':test_acc,
                'test_f1':test_f1,
                'val_f1':best_f1_score,
                'timestamp':timestr}
    
    dict_multiple_seeds['run_'+str(i)]=dict_temp

#obtain the timestring here 
end_time_str=time.strftime("%Y%m%d-%H%M%S")

#dictionary file 
destination_file=os.path.join(destination_run_folder,'multi_run_'+option+'_'+end_time_str+'.json')

#save the dictionary to the json file
with open(destination_file, 'w') as fp:
    json.dump(dict_multiple_seeds, fp, indent=4)




