import os 
import pandas as pd 
import argparse
import whisper
import torch
from tqdm import tqdm 

#argparse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='large')
parser.add_argument('--download_root_folder', type=str, help='Download path root for the model weights')
parser.add_argument('--source_folder',type=str,help='Source folder for the audio files')
parser.add_argument('--save_folder',type=str,help='Save folder for the extracted transcripts')

#command line arguments 
args = parser.parse_args()
model_name = args.model_name
download_root_folder = args.download_root_folder
source_folder=args.source_folder
save_folder=args.save_folder

#load the model
model = whisper.load_model(model_name, download_root=download_root_folder)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

wav_file_names=os.listdir(source_folder)
wav_file_names=[os.path.join(source_folder,i) for i in wav_file_names]

for wav_file in tqdm(wav_file_names):
    file_key=os.path.splitext(wav_file.split("/")[-1])[0]
    save_file_name=file_key+".txt"
    save_file_path=os.path.join(save_folder,save_file_name)

    if (os.path.exists(save_file_path) is False):
        result = model.transcribe(wav_file)

        #save the text in a specific file 

        file_key=os.path.splitext(wav_file.split("/")[-1])[0]
        save_file_name=file_key+".txt"

        save_file_path=os.path.join(save_folder,save_file_name)

        with open(save_file_path, 'w') as f:
            f.write(result['text'])

        #print(result['text'])

