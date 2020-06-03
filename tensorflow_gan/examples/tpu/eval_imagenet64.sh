
export EXPERIMENT_NAME=imagenet64_fixed
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=64 \
--dataset_name=imagenet_resized/64x64 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=acgan \
--num_eval_steps=1 \
--mode=intra_fid_eval'


bash tpu/_eval_base.sh