
export EXPERIMENT_NAME=imagenet128_expt_trial2_bkp
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=128 \
--dataset_name=imagenet2012 \
--num_classes=1000 \
--dataset_val_split_name=train'
export ADDITIONAL="--critic_type=acgan \
--num_eval_steps=40 \
--mode=intra_fid_eval \
--intra_fid_eval_chunk_size=20"

bash tpu/_eval_base.sh

