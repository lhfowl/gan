
export EXPERIMENT_NAME=imagenet64_baseline_ctd
export BATCH_SIZE=1024
export TRAIN_STEPS_PER_EVAL=2500
export DATASET_ARGS='--image_size=64 \
--dataset_name=imagenet_resized/64x64 \
--num_classes=1000 \
--dataset_val_split_name=validation'
export ADDITIONAL='--critic_type=acgan_multiproj \
--aux_mhinge_cond_generator_weight=0.1 \
--aux_mhinge_cond_discriminator_weight=1.0 \
--extra_eval_metrics'


bash tpu/_base.sh