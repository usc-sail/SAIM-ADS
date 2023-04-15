import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle

#append path of datasets and models 
sys.path.append(os.path.join('..', '..', '..','datasets'))
sys.path.append(os.path.join('..', '..', '..','models'))
sys.path.append(os.path.join('..', '..', '..','configs'))
sys.path.append(os.path.join('..', '..', '..','losses'))
sys.path.append(os.path.join('..', '..','..','optimizers'))
sys.path.append(os.path.join('..', '..','..','utils'))
sys.path.append(os.path.join('..', '..'))
#import all libraries 
import random
from ast import literal_eval
import torch
import yaml
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from multi_task_dataset import *
from loss_functions import *
from MHA_models import *
from optimizer import *
from metrics import calculate_stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
from evaluate_multi_task_model import *
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
import json
from statistics import mean
import numpy as np 
from transformers import BertTokenizer, BertModel, BertConfig

config_file=""

#load the config file
with open(config_file,'r') as f:
    config_data = yaml.safe_load(f)

csv_file=config_data['data']['csv_file']
topic_file=config_data['data']['topic_file']
csv_data=pd.read_csv(csv_file)
transcript_file=config_data['data']['transcript_file']
base_folder=config_data['data']['base_folder']

#task label map 
task_dict=config_data['parameters']['task_dict']
weight_dict=config_data['loss']['weight_dict']

#create the label maps here 
with open(topic_file,'r') as f:
    topic_dict=json.load(f)

#complete label map for all tasks 
label_map={'Topic':topic_dict,
           'Transition_val':{'No transition':0,'Transition':1},
           'social_message':{'No':0,'Yes':1}}

sampled_label_map={k:label_map[k] for k in task_dict.keys()}

## general parameters 
#parameters regarding number of classes, maximum audio length, maximum video length
max_text_length=config_data['parameters']['text_max_length']
max_video_length=config_data['parameters']['video_max_length']
batch_size=config_data['parameters']['batch_size']
num_epochs=config_data['parameters']['epochs']
num_workers=config_data['parameters']['num_workers']

#parameters regarding the perceiver model
text_dim=config_data['model']['text_dim']
video_dim=config_data['model']['video_dim']
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
multi_run_folder=config_data['output']['multiple_run_folder']
tokenizer=BertTokenizer.from_pretrained(model_name)

option=model_type
weight_comb_name=""
for key in task_dict.keys():
    weight_comb_name=weight_comb_name+"_"+str(weight_dict[key])+"_"+str(key)
option=option+weight_comb_name

#num runs and multi run folder 
num_runs=config_data['parameters']['num_runs']
multi_run_folder=config_data['output']['multiple_run_folder']


#create the model type folder inside the multi run folder
destination_run_folder=os.path.join(multi_run_folder,model_type)
if not os.path.exists(destination_run_folder):
    os.mkdir(destination_run_folder)

#create random seeds for multiple runs 
seed_list = [random.randint(1, 100000) for _ in range(5)]

#test and validation metrics 
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

#dictionary containing the loss functions for all taskss
loss_function_dict={'Topic': multi_class_cross_entropy_loss(device),
                    'Transition_val':binary_cross_entropy_loss(device), #changing to sigmoid focal loss 
                    'social_message':sigmoid_focal_loss}

#sampled dictionary
sampled_loss_function_dict={k:loss_function_dict[k] for k in task_dict.keys()}


#activations dict
activation_dict={'Topic':nn.Softmax(dim=-1),
                 'Transition_val':nn.Sigmoid(),
                'social_message':nn.Sigmoid()}

#sampled activation dict
sampled_activation_dict={k:activation_dict[k] for k in task_dict.keys()}

for i,seed in enumerate(seed_list):

    # global fixing here 
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Run with random seed: %d' %(seed))

    #define the datasets 
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']
    test_data=csv_data[csv_data['Split']=='test']

    train_ds=SAIM_multi_task_dataset_visual_text_shot_level(train_data,
                                                                transcript_file,tokenizer,
                                                                base_folder,sampled_label_map,
                                                                max_text_length,max_video_length)
    val_ds=SAIM_multi_task_dataset_visual_text_shot_level(val_data,
                                                                transcript_file,tokenizer,
                                                                base_folder,sampled_label_map,
                                                                max_text_length,max_video_length)
    test_ds=SAIM_multi_task_dataset_visual_text_shot_level(test_data,
                                                                transcript_file,tokenizer,
                                                                base_folder,sampled_label_map,
                                                                max_text_length,max_video_length)
    #define the dataloaders and datasets
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

    #define the device here
    if(config_data['device']['is_cuda']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model here 
    #define the model
    params_dict={'text_dim':text_dim,
                 'video_dim':video_dim,
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

    model=Perceiver_TextVisual_multi_task_model(**params_dict)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters: %d' %(params))
    model=model.to(device)

    #optimizer declaration here
    if(config_data['optimizer']['choice']=='Adam'):
        optim_example=optimizer_adam(model,float(config_data['optimizer']['lr']))

    elif(config_data['optimizer']['choice']=='AdamW'):
        optim_example=optimizer_adamW(model,float(config_data['optimizer']['lr']),float(config_data['optimizer']['decay']))

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

    best_f1_score=0.0

    for epoch in range(1, num_epochs+1): #main outer loop

        train_loss_dict={k:[] for k in task_dict.keys()}
        step=0
        t = time.time()

        #target labels dict for each task
        target_labels_dict={k:[] for k in task_dict.keys()}
        pred_labels_dict={k:[] for k in task_dict.keys()}

        #predicted labels dict for each task
        pred_labels_np={k:[] for k in task_dict.keys()}
        target_labels_np={k:[] for k in task_dict.keys()}

        #validation loss dict for each task
        val_loss_dict={k:[] for k in task_dict.keys()}
        train_loss_list=[] #overall training loss
        
        #f1 score and accuracy dict for each task
        f1_score_dict={k:[] for k in task_dict.keys()}
        acc_score_dict={k:[] for k in task_dict.keys()}

        for id,return_dict in enumerate(tqdm(train_dl)):

            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            input_ids=return_dict['input_ids'].to(device)
            attention_mask=return_dict['attention_mask'].to(device)
            token_type_ids=return_dict['token_type_ids'].to(device)
            label_dict=return_dict['label']

            #return dict contains video features and attention mask
            video_feat=return_dict['video_feat'].float()

            #print dtype of video_feat
            #print('dtype of video_feat:%s' %(video_feat.dtype))
            video_feat=video_feat.to(device)
            video_attn_mask=return_dict['video_attn_mask'].to(device)

            for task in label_dict.keys():
                label_dict[task]=label_dict[task].to(device)

            optim_example.zero_grad()
            task_logits=model(input_ids=input_ids,
                         visual_inputs=video_feat,
                         text_mask=attention_mask,
                         visual_mask=video_attn_mask)
            

            #loss calculation here
            loss=torch.tensor(0.0).to(device)
            
            #train loss dictionary
            for task in task_dict.keys():
                #print(task_logits[task].shape,label_dict[task].shape,sampled_loss_function_dict[task])
                loss_task=sampled_loss_function_dict[task](task_logits[task],label_dict[task])
                loss+=weight_dict[task]*loss_task
                train_loss_dict[task].append(loss_task.item())

            train_loss_list.append(loss.item())

            #backward pass and optimization step
            loss.backward()
            optim_example.step()

            #obtain the logits here 
            for k in task_dict.keys():
                activation_logits_task=sampled_activation_dict[k](task_logits[k])
                pred_labels_dict[k].append(activation_logits_task.cpu())
                target_labels_dict[k].append(label_dict[k].cpu())

            step+=1

            if(step%150==0):
                logger_step_dict={'Running_Train_loss':mean(train_loss_list)}
                logger.info("Training loss:{:.3f}".format(loss.item()))

        #agrgegate the labels here
        for k in task_dict.keys():
            
            pred_lb_np=torch.cat(pred_labels_dict[k]).detach().numpy()
            if((k=='social_message') or (k=='Transition_val')):
                pred_lb_np=np.where(pred_lb_np>=0.5,1,0)

            else:
                pred_lb_np=np.argmax(pred_lb_np,axis=1)

            pred_labels_np[k]=pred_lb_np
            target_labels_np[k]=torch.cat(target_labels_dict[k]).detach().numpy()    

            #print(target_labels_np[k].shape,pred_labels_np[k].shape)
            f1_score_dict[k]=f1_score(target_labels_np[k],pred_labels_np[k],average='macro')
            acc_score_dict[k]=accuracy_score(target_labels_np[k],pred_labels_np[k])

        logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
        logger.info('Epoch:{:d},Overall Training loss:{:.3f}'.format(epoch,mean(train_loss_list)))

        
        for k in task_dict.keys():
            logger.info('Epoch:{:d},Task:{},Train loss:{:.3f},Train F1 score:{:.3f},Train Acc score:{:.3f}'.format(epoch,k,mean(train_loss_dict[k]),f1_score_dict[k],acc_score_dict[k]))

        logger.info('Evaluating the dataset')
        val_loss,val_loss_dict,val_f1_dict,val_acc_dict=gen_validate_score_text_visual_perceiver_multi_task(model,val_dl,device,task_dict,sampled_activation_dict,sampled_loss_function_dict,weight_dict)

        logger.info('Epoch:{:d},Overall validation loss:{:.3f}'.format(epoch,val_loss))
        for k in task_dict.keys():
            logger.info('Epoch:{:d},Task:{},Val loss:{:.3f}, Val F1 score:{:.3f}, Val Acc score:{:.3f}'.format(epoch,k, mean(val_loss_dict[k]), val_f1_dict[k],val_acc_dict[k]))

        model.train(True)

        #mean over all validation values in the dictionary
        val_f1_score=0.0
        for k in task_dict.keys():
            val_f1_score+=weight_dict[k]*val_f1_dict[k]

        if(val_f1_score>best_f1_score):
            best_f1_score=val_f1_score
            logger.info('Saving the best model')
            logger.info('Best F1 score:{:.3f}'.format(best_f1_score))
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

    #load the best model here
    test_loss,test_loss_dict,test_f1_dict,test_acc_dict=gen_validate_score_text_visual_perceiver_multi_task(model,test_dl,device,task_dict,sampled_activation_dict,sampled_loss_function_dict,weight_dict)
                    
    #calculate the f1 score here
    for k in task_dict.keys():
        logger.info('Epoch:{:d},Task:{},Test loss:{:.3f},Test F1 score:{:.3f}, Test Acc score:{:.3f}'.format(epoch,k, mean(test_loss_dict[k]), test_f1_dict[k],test_acc_dict[k]))

    dict_temp={ 'seed':seed,
                'test_loss':test_loss_dict,
                'test_acc':test_acc_dict,
                'test_f1':test_f1_dict,
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







            









