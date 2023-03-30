import os 
import pandas as pd 
import argparse
import whisper
import torch
from tqdm import tqdm 
import json
import numpy as np

def detect_language(model,audio_file):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    language=max(probs, key=probs.get)
    prob_detect=probs[language]
    
    return (language,prob_detect)

#argparse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='large')
parser.add_argument('--download_root_folder', type=str, help='Download path root for the model weights')
parser.add_argument('--source_file',type=str,help='json file containing the wav file paths')

#command line arguments 
args = parser.parse_args()
model_name = args.model_name
download_root_folder = args.download_root_folder
source_file=args.source_file

#load the model
model = whisper.load_model(model_name, download_root=download_root_folder)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load the json file
json_data=json.load(open(source_file, 'r'))
wav_file_list=[json_data['data'][i]['wav'] for i in np.arange(len(json_data['data']))]
#
# 
# print(wav_file_list)
total_dict={}
num_files=0
for wav_file in tqdm(wav_file_list):

    #detect language
    language,prob=detect_language(model,wav_file)

    #language dictionary
    language_dict={'label':language,'score':prob}

    total_dict[os.path.splitext(wav_file.split("/")[-1])[0]]=language_dict

    num_files+=1

    # if(num_files==100):
    #     break

#save the json file
with open('../../data/language_detection_whisper.json', 'w') as fp:
    json.dump(total_dict, fp, indent=4)









