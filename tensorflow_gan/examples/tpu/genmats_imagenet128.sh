export EXPERIMENT_NAME=ssl_imagenet128_withproj_baseline
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=128 \
--dataset_name=imagenet2012 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=acgan \
--mode=gen_matrices'

bash tpu/_eval_base.sh

