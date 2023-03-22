## Caption extraction for middle shots 

* Extract captions using BLIP large model using the following command (hpc):

    ```bash
    CUDA_VISIBLE_DEVICES=0 python shot_sample_caption_extraction.py --caption_folder /scratch1/dbose/ads_repo/captions/csv_files --image_folder /scratch1/dbose/ads_repo/captions/images --model_name blip_caption --model_type large_coco --shot_folder /scratch1/dbose/ads_repo/shot_folder/PySceneDetect --shard_file /project/shrikann_35/dbose/codes/ads_codes/caption_split_file/caption_split_0.txt
    ```