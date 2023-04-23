from tqdm import tqdm 
import numpy as np 
import torch 
from statistics import mean 
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.stats.stats import pearsonr
import sys
import os 
from collections import Counter

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths

def gen_validate_score_LSTM_tone_transition_model(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (vid_feat,label,lens) in enumerate(tqdm(loader)):

            vid_feat=vid_feat.float()
            label=label.float()
            vid_feat=vid_feat.to(device)
            label=label.to(device)

            vid_feat,label,lens = sort_batch(vid_feat,label,lens)
            logits=model(vid_feat,lens.cpu().numpy())
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()
    #print(target_label_val.shape,pred_label_val.shape)
    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)

    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)
    
    #print(len(target_labels),len(pred_labels_discrete))
    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')

    #classification_rep=classification_report(target_label_val,pred_labels_discrete)

    cm=confusion_matrix(list(target_labels_array),list(pred_labels_array),labels=[0,1])


    return(mean(val_loss_list),val_acc,val_f1,cm)

#same as previous... neeed to merge later
def gen_validate_score_LSTM_social_message_model(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (vid_feat,label,lens) in enumerate(tqdm(loader)):

            vid_feat=vid_feat.float()
            label=label.float()
            vid_feat=vid_feat.to(device)
            label=label.to(device)

            vid_feat,label,lens = sort_batch(vid_feat,label,lens)
            logits=model(vid_feat,lens.cpu().numpy())
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()
    
    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)

    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')

    #classification_rep=classification_report(target_label_val,pred_labels_discrete)
    cm=confusion_matrix(list(target_labels_array),list(pred_labels_array),labels=[0,1])
    #print(Counter(list(pred_labels_array)))

    return(mean(val_loss_list),val_acc,val_f1,cm)

def gen_validate_score_LSTM_single_task_soc_message_tone(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (vid_feat,label,lens) in enumerate(tqdm(loader)):

            vid_feat=vid_feat.float()
            label=label.float()
            vid_feat=vid_feat.to(device)
            label=label.to(device)

            vid_feat,label,lens = sort_batch(vid_feat,label,lens)
            logits=model(vid_feat,lens.cpu().numpy())
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()
    
    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)

    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_LSTM_single_task_topic(model,loader,device,criterion):

    print("starting validation")
    _softmax=nn.Softmax(dim=-1)
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (vid_feat,label,lens) in enumerate(tqdm(loader)):

            vid_feat=vid_feat.float()
            vid_feat=vid_feat.to(device)
            label=label.to(device)

            vid_feat,label,lens = sort_batch(vid_feat,label,lens)
            logits=model(vid_feat,lens.cpu().numpy())
            val_logits=_softmax(logits)
            y_pred=torch.max(val_logits, 1)[1]


            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()
    
    #convert pred_labels_discrete to 0 and 1 using argmax
    #pred_labels_array=np.argmax(pred_labels,axis=1)
    # target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_label_val,pred_label_val)
    val_f1=f1_score(target_label_val,pred_label_val,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_MHA_model_single_task_soc_message_tone(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (vid_feat,label,mask) in enumerate(tqdm(loader)):

            vid_feat=vid_feat.float()
            vid_feat=vid_feat.to(device)
            label=label.to(device)

            mask=mask.unsqueeze(1).unsqueeze(1).to(device)
            logits=model(vid_feat,mask=mask)
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()

    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)
    
    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)


def gen_validate_score_MHA_model_single_task_topic(model,loader,device,criterion):

    print("starting validation")
    log_softmax=nn.LogSoftmax(dim=-1)
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (feat,label,mask) in enumerate(tqdm(loader)):

            feat=feat.float()
            feat=feat.to(device)
            label=label.to(device)

            mask=mask.unsqueeze(1).unsqueeze(1).to(device)
            logits=model(feat,mask=mask)
            logits=log_softmax(logits)
            y_pred=torch.max(logits, 1)[1]

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).detach().numpy()
    pred_label_val=torch.cat(pred_labels).detach().numpy()

    #accuracy and f1 score for topic
    val_acc=accuracy_score(target_label_val,pred_label_val)
    val_f1=f1_score(target_label_val,pred_label_val,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_perceiver_single_task_soc_message_tone(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():

        for i, (audio_feat,video_feat,label,audio_mask,video_mask) in enumerate(tqdm(loader)):

            audio_feat=audio_feat.float()
            audio_feat=audio_feat.to(device)
            video_feat=video_feat.float()
            video_feat=video_feat.to(device)
            label=label.to(device)
            audio_mask=audio_mask.bool()
            video_mask=video_mask.bool()
            audio_mask=audio_mask.to(device)
            video_mask=video_mask.to(device)

            logits=model(audio_inputs=audio_feat,
                visual_inputs=video_feat,
                audio_mask=audio_mask,
                visual_mask=video_mask)
            
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()

    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)
    
    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_perceiver_single_task_topic(model,loader,device,criterion):

    print("starting validation")
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]
    log_softmax=nn.LogSoftmax(dim=-1)

    with torch.no_grad():

        for i, (audio_feat,video_feat,label,audio_mask,video_mask) in enumerate(tqdm(loader)):
            
            #audio features
            audio_feat=audio_feat.float()
            audio_feat=audio_feat.to(device)

            #video features
            video_feat=video_feat.float()
            video_feat=video_feat.to(device)

            #label
            label=label.to(device)

            #audio and video mask
            audio_mask=audio_mask.bool()
            video_mask=video_mask.bool()
            audio_mask=audio_mask.to(device)
            video_mask=video_mask.to(device)

            logits=model(audio_inputs=audio_feat,
                visual_inputs=video_feat,
                audio_mask=audio_mask,
                visual_mask=video_mask)
            
            logits=log_softmax(logits)
            y_pred=torch.max(logits, 1)[1]

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).detach().numpy()
    pred_label_val=torch.cat(pred_labels).detach().numpy()

    #accuracy and f1 score for topic
    val_acc=accuracy_score(target_label_val,pred_label_val)
    val_f1=f1_score(target_label_val,pred_label_val,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)
            
def gen_validate_score_text_visual_perceiver_single_task_soc_message_tone(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]
    logits_list=[]
    clip_keys_list=[]

    with torch.no_grad():

        for id,(return_dict) in enumerate(tqdm(loader)):

            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            input_ids=return_dict['input_ids'].to(device)
            attention_mask=return_dict['attention_mask'].to(device)
            token_type_ids=return_dict['token_type_ids'].to(device)

            #return dict contains video features and attention mask
            video_feat=return_dict['video_feat'].float()
            video_feat=video_feat.to(device)
            video_attn_mask=return_dict['video_attn_mask'].to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            logits=model(input_ids=input_ids,
                         visual_inputs=video_feat,
                         text_mask=attention_mask,
                         visual_mask=video_attn_mask)
            
            logits_list.append(logits)
            
            logits_sig=Sig(logits)

            clip_keys_list.append(return_dict['clip_key'])

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()

    logits_array=torch.cat(logits_list).detach().cpu().numpy()

    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)

    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')

    return(mean(val_loss_list),val_acc,val_f1,logits_array,clip_keys_list)

def gen_validate_score_text_visual_perceiver_single_task_topic(model,loader,device,criterion):

    print("starting validation")
    
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]
    logits_list=[]
    clip_keys_list=[]
    log_softmax=nn.Softmax(dim=-1)

    with torch.no_grad():

        for id,(return_dict) in enumerate(tqdm(loader)):

            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            input_ids=return_dict['input_ids'].to(device)
            attention_mask=return_dict['attention_mask'].to(device)
            token_type_ids=return_dict['token_type_ids'].to(device)

            #return dict contains video features and attention mask
            video_feat=return_dict['video_feat'].float()
            video_feat=video_feat.to(device)
            video_attn_mask=return_dict['video_attn_mask'].to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            logits=model(input_ids=input_ids,
                         visual_inputs=video_feat,
                         text_mask=attention_mask,
                         visual_mask=video_attn_mask)
            
            logits_list.append(logits)
            clip_keys_list.extend(return_dict['clip_key'])
            
            logits=log_softmax(logits)
            y_pred=torch.max(logits, 1)[1]

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).detach().numpy()
    pred_label_val=torch.cat(pred_labels).detach().numpy()

    logits_array=torch.cat(logits_list).detach().cpu().numpy()

    #accuracy and f1 score for topic
    val_acc=accuracy_score(target_label_val,pred_label_val)
    val_f1=f1_score(target_label_val,pred_label_val,average='macro')

    return(mean(val_loss_list),val_acc,val_f1,logits_array,clip_keys_list)

def gen_validate_score_SBERT_text_visual_perceiver_single_task_soc_message_tone(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():

        for id,(return_dict) in enumerate(tqdm(loader)):

            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            text_feat=return_dict['text_feat'].float()
            text_feat=text_feat.to(device)

            attention_mask=return_dict['attention_mask'].to(device)
            video_attn_mask=return_dict['video_attn_mask'].to(device)

            #return dict contains video features and attention mask
            video_feat=return_dict['video_feat'].float()
            video_feat=video_feat.to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            logits=model(text_inputs=text_feat,
                         visual_inputs=video_feat,
                         text_mask=attention_mask,
                         visual_mask=video_attn_mask)
            
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()

    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)

    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')


    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_SBERT_text_visual_perceiver_single_task_topic(model,loader,device,criterion):

    print("starting validation")
    
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]
    log_softmax=nn.LogSoftmax(dim=-1)


    with torch.no_grad():

        for id,(return_dict) in enumerate(tqdm(loader)):

            text_feat=return_dict['text_feat'].float()
            text_feat=text_feat.to(device)

            attention_mask=return_dict['attention_mask'].to(device)
            video_attn_mask=return_dict['video_attn_mask'].to(device)

            #return dict contains video features and attention mask
            video_feat=return_dict['video_feat'].float()
            video_feat=video_feat.to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            logits=model(text_inputs=text_feat,
                         visual_inputs=video_feat,
                         text_mask=attention_mask,
                         visual_mask=video_attn_mask)

            #loss calculation here
            loss = criterion(logits, label)
            logits=log_softmax(logits)
            y_pred=torch.max(logits, 1)[1]

            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).detach().numpy()
    pred_label_val=torch.cat(pred_labels).detach().numpy()

    #accuracy and f1 score for topic
    val_acc=accuracy_score(target_label_val,pred_label_val)
    val_f1=f1_score(target_label_val,pred_label_val,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_audio_text_perceiver_single_task_soc_message_tone(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():

        for id,(return_dict) in enumerate(tqdm(loader)):

            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            input_ids=return_dict['input_ids'].to(device)
            attention_mask=return_dict['attention_mask'].to(device)
            token_type_ids=return_dict['token_type_ids'].to(device)

            #return dict contains video features and attention mask
            audio_feat=return_dict['audio_feat'].float()

            #print dtype of video_feat
            #print('dtype of video_feat:%s' %(video_feat.dtype))

            audio_feat=audio_feat.to(device)
            audio_attn_mask=return_dict['audio_attn_mask'].to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            logits=model(input_ids=input_ids,
                         audio_inputs=audio_feat,
                         text_mask=attention_mask,
                         audio_mask=audio_attn_mask)

            #loss calculation here
            loss = criterion(logits, label)
            logits_sig=Sig(logits)

            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()

    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)

    #convert pred_labels_discrete to 0 and 1 using argmax
    pred_labels_array=np.argmax(pred_labels_discrete,axis=1)
    target_labels_array=np.argmax(target_label_val,axis=1)

    val_acc=accuracy_score(target_labels_array,pred_labels_array)
    val_f1=f1_score(target_labels_array,pred_labels_array,average='macro')


    return(mean(val_loss_list),val_acc,val_f1)

def gen_validate_score_audio_text_perceiver_single_task_topic(model,loader,device,criterion):

    print("starting validation")
    log_softmax= nn.Softmax(dim=-1)
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():

        for id,(return_dict) in enumerate(tqdm(loader)):

            # return dict contains the following keys: input ids, attention_maksk, token_type_ids
            input_ids=return_dict['input_ids'].to(device)
            attention_mask=return_dict['attention_mask'].to(device)
            token_type_ids=return_dict['token_type_ids'].to(device)

            #return dict contains video features and attention mask
            audio_feat=return_dict['audio_feat'].float()

            #print dtype of video_feat
            #print('dtype of video_feat:%s' %(video_feat.dtype))

            audio_feat=audio_feat.to(device)
            audio_attn_mask=return_dict['audio_attn_mask'].to(device)

            #return dict contains labels
            label=return_dict['label'].to(device)

            logits=model(input_ids=input_ids,
                         audio_inputs=audio_feat,
                         text_mask=attention_mask,
                         audio_mask=audio_attn_mask)

            #loss calculation here
            loss = criterion(logits, label)
            val_logits=log_softmax(logits)
            y_pred=torch.max(val_logits, 1)[1]

            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(y_pred.cpu())
            step=step+1

    target_label_np=torch.cat(target_labels).detach().numpy()
    pred_label_np=torch.cat(pred_labels).detach().numpy()

    val_acc=accuracy_score(target_label_np,pred_label_np)
    val_f1=f1_score(target_label_np,pred_label_np,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)















            






