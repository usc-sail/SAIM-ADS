import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio 
import torch

file="/data/digbose92/ads_complete_repo/ads_wav_files/jwt_ads_of_world_wav_files/459528.wav"
waveform,sampling_rate=torchaudio.load(file) 
waveform=waveform[0].cpu().numpy()

waveform = torch.from_numpy(waveform).unsqueeze(0)
num_mel_bins=128
max_length=1024
fbank = ta_kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=10
)

n_frames = fbank.shape[0]
difference = max_length - n_frames
print(n_frames,fbank.shape)
# # pad or truncate, depending on difference
# if difference > 0:
#     pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
#     fbank = pad_module(fbank)
# elif difference < 0:
#     fbank = fbank[0:max_length, :]
#     fbank = fbank.numpy()

# print(fbank.shape)