# FAKEOUT TO GET MATRICES
export EXPERIMENT_NAME=imagenet64_baseline_ctd
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=128 \
--dataset_name=imagenet2012 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=acgan \
--num_eval_steps=49 \
--mode=gen_images \
--extra_eval_metrics'

bash tpu/_eval_base.sh

