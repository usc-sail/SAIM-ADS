## Instructions for training and testing MHA models


### Training

* To train a model for topic classification, run the following command:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python train_MHA_model_topic_class.py --config_file ../../config_MHA_topic_classifier.json
    ```

* To train the model for social message presence/absence or tone transition run the following command:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python train_MHA_model_soc_msg_tone.py --config_file ../../config_MHA_single_task_classifier.json
    ```

### Testing 

* To test the model for topic classification, run the following command:

    ```bash 
    CUDA_VISIBLE_DEVICES=0 python test_MHA_model_topic_class.py --config_file /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/log_dir/MHA_attn_single_task_classifier_Topic/20230322-175130_MHA_attn_single_task_classifier_Topic/20230322-175130_MHA_attn_single_task_classifier_Topic.yaml --model_file /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/model_dir/MHA_attn_single_task_classifier_Topic/20230322-175130_MHA_attn_single_task_classifier_Topic/20230322-175130_MHA_attn_single_task_classifier_Topic_best_model.pt
    ```

* To test the model for social message presence/absence or tone transition run the following command:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_MHA_model_soc_msg_tone.py --config_file /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/log_dir/MHA_attn_single_task_classifier_social_message/20230323-160409_MHA_attn_single_task_classifier_social_message/20230323-160409_MHA_attn_single_task_classifier_social_message.yaml --model_file /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/model_dir/MHA_attn_single_task_classifier_social_message/20230323-160409_MHA_attn_single_task_classifier_social_message/20230323-160409_MHA_attn_single_task_classifier_social_message_best_model.pt
    ```

