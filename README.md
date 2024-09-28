# SAIM-ADS
Repository for experiments and preprocessing related to advertisement videos analysis

## Environment creation:

* Create a conda environment using the following command:

```bash
conda create -n ads-env python=3.8
```

* For creating conda enviroment with specific path:

```bash
conda create --prefix <path> python=3.8
```

* Install pytorch using the following command:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
* Install additional requirements using the following:

```
pip install -r requirements.txt
```

## Shot Extraction

* Install PySceneDetect using the following command:

```bash
pip install scenedetect[opencv] --upgrade
```

## Install CLIP 

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Trancripts extraction 

### Extracting transcripts using Whisper 

* Install whisper using the following command:
   ```bash
   pip install -U openai-whisper
   ```
* While instantiating the model provide the download root path for the model:
   ```python 
      import whisper 
      model=whisper.load_model("large", download_root="path/to/download/model")
   ```

### Extracting transcripts using whisper-X 

* Follow the instructions listed in [Whisper-X](https://github.com/m-bain/whisperX) for installation:

   ```
   pip install git+https://github.com/m-bain/whisperx.git
   ```

## Feature extraction 

* Go to folder feature_extraction and for extracting shot level features using vision transformers use the following 

   ```bash
   CUDA_VISIBLE_DEVICES=3 python extract_vit_features.py --feature_folder <destination vit features> --video_folder <base folder containing the shots> --   
    model_name google/vit-base-patch16-224 --video_type shot --shot_subfolder <type of shot here>
   ```

* The previous command has been modified to extract features for files that have not been processed in the list of files provided in the file_list.txt file.

   ```
   CUDA_VISIBLE_DEVICES=3 python extract_vit_features.py --feature_folder <destination vit features>  --video_folder <base folder containing the shots> --model_name google/vit-base-patch16-224 --video_type shot --shot_subfolder <type of shot here> --shot_file_list <path to file list>
   ```

## Visual caption extraction 

* Create a new conda environment using the following:

   ```bash 
   conda create -n lavis python=3.8
   conda activate lavis
   ```
* Install lavis using the following: 

   ```
   pip install salesforce-lavis
   ```

Associated citation:

@article{Bose2023MMAUTowardsMU,
  title={MM-AU: Towards Multimodal Understanding of Advertisement Videos},
  author={Digbalay Bose and Rajat Hebbar and Tiantian Feng and Krishna Somandepalli and Anfeng Xu and Shrikanth S. Narayanan},
  journal={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023},
  url={https://dl-acm-org.libproxy2.usc.edu/doi/10.1145/3581783.3612371}
}








