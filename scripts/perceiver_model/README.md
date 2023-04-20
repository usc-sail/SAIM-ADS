## Sample instructions 

* Run the following instructions to train the model on the SAIM-ADS dataset for social message task.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_multi_seeds_text_shot_level_perceiver_model.py --log_dir /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/multi_run_log_dir --model_dir /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/multi_run_model_dir --folder_name Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message --json_file /data/digbose92/ads_complete_repo/ads_codes/model_files/multi_run_folder/Perceiver_single_task_classifier_shot_level_multiple_seeds/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230409-154642.json
    ```

* Run the following instructions to test the model on the SAIM-ADS dataset for transition task.

    ```bash
    CUDA_VISIBLE_DEVICES=2 python test_multi_seeds_text_shot_level_perceiver_model.py --log_dir /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/multi_run_log_dir --model_dir /data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/multi_run_model_dir --folder_name Perceiver_single_task_classifier_shot_level_multiple_seeds_Transition_val --json_file /data/digbose92/ads_complete_repo/ads_codes/model_files/multi_run_folder/Perceiver_single_task_classifier_shot_level_multiple_seeds/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_Transition_val_20230409-035050.json
```
