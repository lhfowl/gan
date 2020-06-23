
# export EXPERIMENT_NAME=imagenet64
# export BATCH_SIZE=1024
# export TRAIN_STEPS_PER_EVAL=4000
# export DATASET_ARGS='--image_size=64 \
# --dataset_name=imagenet_resized/64x64 \
# --num_classes=1000 \
# --dataset_val_split_name=validation'
# export ADDITIONAL='--critic_type=acgan \
# --aux_mhinge_cond_generator_weight=1.0 \
# --aux_mhinge_cond_discriminator_weight=0.1'

export EXPERIMENT_NAME=imagenet64_baseline
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=10000
export DATASET_ARGS='--image_size=64 --dataset_name=imagenet_resized/64x64 --num_classes=1000 --dataset_val_split_name=validation'

export ADDITIONAL='--critic_type=acgan \
--extra_eval_metrics=true \
--keep_checkpoint_max=10'

bash tpu/_base.sh