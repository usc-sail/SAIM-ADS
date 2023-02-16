# SAIM-ADS
Repository for experiments and preprocessing related to advertisement videos analysis

## Associated repository listing:

Please find links of associated repositories listed here:
### Repository for annotation framework:

* [Ads Mturk framework designs](https://github.com/usc-sail/mica-ads-Mturk-experiments)

### Repository for listing previous experiments (topic understanding, data parsing):

* [mica ads experiments](https://github.com/usc-sail/mica-ads-experiments)

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
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```
* Install additional requirements using the following:

```
pip install -r requirements.txt
```

## Extracting transcripts using whisper-X 

* Follow the instructions listed in [Whisper-X](https://github.com/m-bain/whisperX) for installation:

   ```
   pip install git+https://github.com/m-bain/whisperx.git
   ```

## TODOS

* Check CLIP features statistics for the video frames 
* LSTM on the CLIP features (variable length) and MHA baselines
* MLP baselines on the CLIP features 









