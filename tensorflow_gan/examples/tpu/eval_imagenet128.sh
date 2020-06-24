
# export EXPERIMENT_NAME=imagenet128_expt_2ctd
# export EXPERIMENT_NAME=imagenet128_expt_trial2_bkp
# export EXPERIMENT_NAME=imagenet128_expt_3ctd_bkp795
export EXPERIMENT_NAME=ssl_imagenet128_withproj
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=128 \
--dataset_name=imagenet2012 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=acgan \
--num_eval_steps=49 \
--mode=continuous_eval \
--extra_eval_metrics'

# export ADDITIONAL='--critic_type=acgan_multiproj \
# --num_eval_steps=49 \
# --mode=continuous_eval \
# --extra_eval_metrics'

bash tpu/_eval_base.sh

