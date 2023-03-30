#this script uses whisper-X to extract transcripts from audio files 
import whisperx
from tqdm import tqdm
import os
import argparse
device = "cuda" 
model = whisperx.load_model("large", device)

args=argparse.ArgumentParser()
args.add_argument("--shard_file",type=str)
args.add_argument("--dest_dir",type=str, default="/scratch2/dbose/ads_complete_dir/ads_wav_files/jwt_ads_of_world_transcripts/whisper_aligned_medium")
args.add_argument("--model_option",type=str, default="medium")
args.add_argument("--align_model_string",type=str, default="WAV2VEC2_ASR_LARGE_LV60K_960H")
args.add_argument("--verbose" ,type=bool, default=False)
#num threads
args.add_argument("--num_threads",type=int, default=8)

#jwt and ads of the world wav files 
#source_dir="/scratch2/dbose/ads_complete_dir/ads_wav_files/jwt_ads_of_world_wav_files"


#parse the arguments
args=args.parse_args()
shard_file=args.shard_file
dest_dir=args.dest_dir
model_option=args.model_option
align_model_string=args.align_model_string
num_threads=args.num_threads

#read the shard file
with open(shard_file) as f:
    source_dir_list=f.readlines()
file_list=[x.strip().split("\n")[0] for x in source_dir_list]

#file_list=os.listdir(source_dir)

for file in tqdm(file_list):
    #result = model.transcribe(wav_file)
    #audio_basename = os.path.basename(wav_file)

    command="whisperx "+file+" --model "+model_option+" --output_dir "+dest_dir+" --align_model "+align_model_string+" --verbose False --align_extend 2 --threads "+str(num_threads)
    os.system(command)
    # print(audio_basename)

    # # load alignment model and metadata
    # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    # # align whisper output
    # result_aligned = whisperx.align(result["segments"], model_a, metadata, wav_file, device)

    # print(result_aligned["segments"])




