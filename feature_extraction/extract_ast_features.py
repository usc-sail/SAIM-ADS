#use transformers library to extract features from ASTs
from transformers import AutoProcessor, ASTModel, AutoFeatureExtractor
import torch
from datasets import load_dataset
import torchaudio
import json
from tqdm import tqdm 
import os 
import pickle

def generate_file_list(json_data,folder):

    wav_file_names=[folder+"/"+i.split("/")[-1] for i in json_data]
    return wav_file_names

#load the model
model_option="MIT/ast-finetuned-audioset-10-10-0.4593"
wav_file_list="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/jwt_ads_of_world_wav_files.json"
folder="/data/digbose92/ads_complete_repo/ads_wav_files/cvpr_wav_files"
option="cvpr_ads"
save_folder="/data/digbose92/ads_complete_repo/ads_features/audio_embeddings/ast_embeddings/cvpr_ads"
#save_folder="/data/digbose92/ads_complete_repo/ads_features/audio_embeddings/ast_embeddings/jwt_ads_of_world"

if(option=="jwt_ads_of_world"):
    with open(wav_file_list) as f:
        wav_file_list = json.load(f)

    wav_file_names_json_data=[wav_file_list["data"][i]["wav"] for i in range(len(wav_file_list['data']))]
    wav_file_names=generate_file_list(wav_file_names_json_data,folder)

elif(option=="cvpr_ads"):

    wav_file_names=os.listdir(folder)
    wav_file_names=[os.path.join(folder,i) for i in wav_file_names]
#print(len(wav_file_names))
#print(wav_file_names)

# wav_file="/data/digbose92/ads_complete_repo/ads_wav_files/jwt_ads_of_world_wav_files/2k_sports_never_say_never_1.wav"

#define feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_option)
#print(feature_extractor.max_length)
model = ASTModel.from_pretrained(model_option)
device=torch.device("cuda:0")
model.to(device)
sampling_rate=16000
file_list_failure=[]

for wav_file in tqdm(wav_file_names):
    try:
        waveform, sampling_rate = torchaudio.load(wav_file) #read the audio using torchaudio
        #print(wav_file)
        waveform=waveform[0].cpu().numpy()
        inputs=feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt") #extract features using transformers
        inputs['input_values']=inputs['input_values'].to(device)
        #print(inputs.keys())
        with torch.no_grad():
            outputs=model(**inputs)

        last_hidden_state=outputs.last_hidden_state
        pooler_output=outputs.pooler_output

        #create dictionary to save 
        save_dict={'last_hidden_state':last_hidden_state.cpu().numpy(),'pooler_output':pooler_output.cpu().numpy()}

        file_name_id=os.path.splitext(wav_file.split("/")[-1])[0]+".pkl"
        destination_filename=os.path.join(save_folder,file_name_id)

        with open(destination_filename, 'wb') as f:
            pickle.dump(save_dict, f)
        #dict_filename=os.path.join(save_folder,wav_file.split("/")[-1]+".npy")
    except:
        file_list_failure.append(wav_file)
        pass 

    

    #create the save file name

    #save_file_name=wav_file.split("/")[-1].split(".")[0]+".npy"



    #the sequence length is 1214 
    #because the spectrogram is 128*1024 which is broken down as follows: (128-16)//10+1=12 and (1024-16)//10+1=101 and 101*12=1212
    #adding two more tokens will make it 1214 which is two CLS tokens
    #print(last_hidden_state.shape)


# waveform, sampling_rate = torchaudio.load(wav_file)
# waveform=waveform[0].cpu().numpy()
# #model option and model
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_option)
# model = ASTModel.from_pretrained(model_option)
# sampling_rate=16000





# # # #generate the datasets
# # dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# # dataset = dataset.sort("id")
# # print(type(dataset[0]["audio"]["array"]))
# # #read the audio file 
# inputs = feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")


# #inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

# #generate outputs 
# with torch.no_grad():
#     outputs=model(**inputs)

# print(outputs.keys())
# # #last hidden state
# last_hidden_state=outputs.last_hidden_state
# print(last_hidden_state.shape)

