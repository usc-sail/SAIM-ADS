#this script uses whisper-X to extract transcripts from audio files 
import whisperx
from tqdm import tqdm
import os
device = "cuda" 
model = whisperx.load_model("large", device)

#jwt and ads of the world wav files 
source_dir="/data/digbose92/ads_complete_repo/ads_wav_files/jwt_ads_of_world_wav_files"
dest_dir="/data/digbose92/ads_complete_repo/ads_transcripts/whisper-aligned/jwt_ads_of_the_world_medium"
model_option="medium"
align_model_string="WAV2VEC2_ASR_LARGE_LV60K_960H"
num_threads=8


file_list=os.listdir(source_dir)

for file in tqdm(file_list):

    wav_file=os.path.join(source_dir,file)
    #result = model.transcribe(wav_file)
    #audio_basename = os.path.basename(wav_file)

    command="whisperx "+wav_file+" --model "+model_option+" --output_dir "+dest_dir+" --align_model "+align_model_string+" --align_extend 2 --threads "+str(num_threads)
    os.system(command)
    # print(audio_basename)

    # # load alignment model and metadata
    # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    # # align whisper output
    # result_aligned = whisperx.align(result["segments"], model_a, metadata, wav_file, device)

    # print(result_aligned["segments"])




