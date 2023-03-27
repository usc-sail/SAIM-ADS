# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import os
import ast
import pickle
import sys
import torch
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import models
import numpy as np
import json
import torchaudio
from tqdm import tqdm 

data_train = "/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/jwt_ads_of_world_cvpr_wav_files_lomond.json"
overlap_perc = 0.25
out_file = '/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs.pkl'

assert 0 <= overlap_perc <= 1, "invalid overlap percentage"

norm_stats = [-6.5773025, 3.9219317]
target_length = 1024
step_size = int((1-overlap_perc)*target_length)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, audio_conf):

        with open(data_json, 'r') as f:
            self.data = json.load(f)['data']

        self.audio_conf = audio_conf 
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
    
    def __getitem__(self, index):

            datum = self.data[index]['wav']

            waveform, sr = torchaudio.load(datum)
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, 
                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
            
            n_frames = fbank.shape[0]
            p = target_length - n_frames #difference between target length and actual length

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)

            fname = datum.split('/')[-1].split('.wav')[0]
            fbank = (fbank - self.norm_mean) / (self.norm_std*2)

            return fname, fbank

    def __len__(self):
        return len(self.data)


class Args:
    def __init__(self, data_train):
        self.data_train = data_train
        self.fstride = 10
        self.tstride = 10
        self.n_class = 527
        self.batch_size = 32
        self.num_workers = 4
        self.imagenet_pretrain = True
        self.audioset_pretrain = True

args = Args(data_train)#, out_dir)
    
audio_conf = {'num_mel_bins': 128, 'mean':norm_stats[0], 'std':norm_stats[1]}

audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')


data_loader = torch.utils.data.DataLoader(
                AudioDataset(args.data_train, audio_conf=audio_conf), batch_size=1, ## BS needs to be 1 here for overlapped inference
                                        shuffle=True, num_workers=args.num_workers, pin_memory=False)

audio_model = torch.nn.DataParallel(audio_model)
audio_model = audio_model.to('cuda')
audio_model.eval()

post_data = {}
emb_data = {}
file_num=0

with torch.no_grad():
    for fnames, audio_input in tqdm(data_loader):

        #print(audio_input.shape)
        #print(fnames[0],audio_input.shape)
        audio_input = audio_input.squeeze(0).unfold(dimension=0, size=target_length, step=step_size).transpose(-1, -2)

        

        audio_input = audio_input.to('cuda')
       
        embeddings = []
        posteriors = []
        for i in range(0, len(audio_input), args.batch_size):
            audio_out, embs = audio_model(audio_input[i: i + args.batch_size])
            embeddings.extend(embs.to('cpu').detach())
            posteriors.extend(audio_out.to('cpu').detach())
        
        emb_data[fnames[0]] = torch.stack(embeddings)
        post_data[fnames[0]] = torch.stack(posteriors)

        file_num+=1
        # if file_num%100==0:
        #     break
        
with open(out_file, 'wb') as f:
    pickle.dump({'data': {'posteriors':post_data, 'embeddings': emb_data}}, f)



