import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle

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
import wandb

#fix seed for reproducibility
seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

def main(config_data):

    csv_file=config_data['data']['csv_file']
    embedding_file=config_data['data']['embedding_file']
    csv_data=pd.read_csv(csv_file)
    task_name=config_data['parameters']['task_name']

    #define the datasets 
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']

    if(task_name=='Transition_val'):
        label_map={'No transition':0,'Transition':1}

    elif(task_name=='social_message'):
        label_map={'No':0,'Yes':1}

    #parameters regarding number of classes, maximum audio length, maximum video length
    max_audio_length=config_data['parameters']['audio_max_length']
    max_video_length=config_data['parameters']['video_max_length']
    batch_size=config_data['parameters']['batch_size']
    num_epochs=config_data['parameters']['epochs']
    num_workers=config_data['parameters']['num_workers']

    #parameters regarding the perceiver model
    audio_dim=config_data['model']['audio_dim']
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
    model_type=config_data['model']['model_type']
    option=model_type+"_"+config_data['parameters']['task_name']

    #Perceiver model declaration
    model=Perceiver_AudioVisual_Model(audio_dim,video_dim,dim,
                            queries_dim,
                            n_classes,depth,
                            num_latents,cross_heads,
                            latent_heads,cross_dim_head,
                            latent_dim_head,latent_dim,
                            weight_tie_layers,seq_dropout_prob,
                            use_queries)
    
    #define the device here
    if(config_data['device']['is_cuda']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model parameters 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters: %d' %(params))
    model=model.to(device)

    #datasets and dataloaders
    train_ds=SAIM_single_task_dataset_audio_visual(csv_data=train_data,
                                                embedding_file=embedding_file,
                                                label_map=label_map,
                                                num_classes=n_classes,
                                                audio_max_length=max_audio_length,
                                                video_max_length=max_video_length,
                                                task_name=task_name)


    val_ds=SAIM_single_task_dataset_audio_visual(csv_data=val_data,
                                                embedding_file=embedding_file,
                                                label_map=label_map,
                                                num_classes=n_classes,
                                                audio_max_length=max_audio_length,
                                                video_max_length=max_video_length,
                                                task_name=task_name)
    
    #define the dataloaders
    train_dl=DataLoader(train_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['train_shuffle'],
                        num_workers=num_workers)

    val_dl=DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['val_shuffle'],
                        num_workers=num_workers)
    
    ############################# loss function + optimizers definition here ################################
    if(config_data['loss']['loss_option']=='bce_cross_entropy_loss'):
        criterion = binary_cross_entropy_loss(device,pos_weights=None)

    # elif(config_data['loss']['loss_option']=='cross_entropy_loss'):
    #     criterion=multi_class_cross_entropy_loss(device)

    if(config_data['optimizer']['choice']=='Adam'):
        optim_example=optimizer_adam(model,float(config_data['optimizer']['lr']))

    ################################ scheduler definition here ################################
    # if(config_data['optimizer']['scheduler']=='step_lr_plateau'):
    #     lr_scheduler=reducelr_plateau(optim_example,mode=config_data['optimizer']['mode'],factor=config_data['optimizer']['factor'],patience=config_data['optimizer']['patience'],
    #     verbose=config_data['optimizer']['verbose'])

    # if(config_data['optimizer']['scheduler']=='step_lr'):
    #     lr_scheduler=steplr_scheduler(optim_example,
    #                     step_size=config_data['optimizer']['step_size'],
    #                     gamma=config_data['optimizer']['gamma'])
        

    #create a folder with each individual model + create a log file for each date time instant
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
    Sig = nn.Sigmoid()
    best_f1_score=0

    for epoch in range(1, num_epochs+1): #main outer loop

        train_loss_list=[]
        train_logits=[]
        step=0
        t = time.time()
        target_labels=[]
        pred_labels=[]
        val_loss_list=[]

        for id,(audio_feat,video_feat,label,audio_mask,video_mask) in enumerate(tqdm(train_dl)):

            audio_feat=audio_feat.to(device)
            audio_feat=audio_feat.float()

            video_feat=video_feat.to(device)
            video_feat=video_feat.float()

            label=label.to(device)

            audio_mask=audio_mask.bool()
            video_mask=video_mask.bool()

            audio_mask=audio_mask.to(device)
            video_mask=video_mask.to(device)

            optim_example.zero_grad()
            logits=model(audio_inputs=audio_feat,
                visual_inputs=video_feat,
                audio_mask=audio_mask,
                visual_mask=video_mask)
            
            #loss calculation here
            loss = criterion(logits, label)
            logits_sig=Sig(logits)

            # Back prop.
            loss.backward()
            optim_example.step()
            train_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())

            step=step+1
            
            if(step%150==0):
                logger_step_dict={'Running_Train_loss':mean(train_loss_list)}
                logger.info("Training loss:{:.3f}".format(loss.item()))
                #wandb.log(logger_step_dict)

        target_label_np=torch.cat(target_labels).detach().numpy()
        pred_label_np=torch.cat(pred_labels).detach().numpy()
        pred_labels_discrete=np.where(pred_label_np>=0.5,1,0)

        #compute training accuracy and F1 score
        train_acc=accuracy_score(target_label_np,pred_labels_discrete)
        train_f1=f1_score(target_label_np,pred_labels_discrete,average='macro')

        logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
        logger.info('Epoch:{:d},Overall Training loss:{:.3f},Overall training Acc:{:.3f}, Overall F1:{:.3f}'.format(epoch,mean(train_loss_list),train_acc,train_f1))

        logger.info('Evaluating the dataset')
        val_loss,val_acc,val_f1=gen_validate_score_perceiver_single_task_soc_message_tone(model,val_dl,device,criterion)
        logger.info('Epoch:{:d},Overall Validation loss:{:.3f},Overall validation Acc:{:.3f}, Overall F1:{:.3f}'.format(epoch,val_loss,val_acc,val_f1))

        #wandb logging
        metrics_dict={'Train_loss':mean(train_loss_list),
            'Train_Acc':train_acc,
            'Train_F1':train_f1,
            'Valid_loss':val_loss,
            'Valid_Acc':val_acc,
            'Valid_corr':val_f1,
            'Epoch':epoch}   #add epoch here to later switch the x-axis with epoch rather than actual wandb log
        
        model.train(True)
        #lr_scheduler.step()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Location of configuration data', type=str, required=True)
    args = vars(parser.parse_args())
    config_data=load_config(args['config_file'])
    main(config_data)  








